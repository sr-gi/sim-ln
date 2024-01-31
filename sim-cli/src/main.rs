use anyhow::anyhow;
use bitcoin::secp256k1::{PublicKey, Secp256k1, SecretKey};
use clap::builder::TypedValueParser;
use clap::Parser;
use log::LevelFilter;
use rand::distributions::Uniform;
use rand::Rng;
use sim_lib::{
    cln::ClnNode,
    lnd::LndNode,
    sim_node::{
        ln_node_from_graph, populate_network_graph, ChannelPolicy, SimGraph, SimulatedChannel,
    },
    ActivityDefinition, LightningError, LightningNode, NodeConnection, NodeId, SimParams,
    Simulation, WriteResults,
};
use simple_logger::SimpleLogger;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// The default directory where the simulation files are stored and where the results will be written to.
pub const DEFAULT_DATA_DIR: &str = ".";

/// The default simulation file to be used by the simulator.
pub const DEFAULT_SIM_FILE: &str = "sim.json";

/// The default expected payment amount for the simulation, around ~$10 at the time of writing.
pub const EXPECTED_PAYMENT_AMOUNT: u64 = 3_800_000;

/// The number of times over each node in the network sends its total deployed capacity in a calendar month.
pub const ACTIVITY_MULTIPLIER: f64 = 2.0;

/// Default batch size to flush result data to disk
const DEFAULT_PRINT_BATCH_SIZE: u32 = 500;

/// Deserializes a f64 as long as it is positive and greater than 0.
fn deserialize_f64_greater_than_zero(x: String) -> Result<f64, String> {
    match x.parse::<f64>() {
        Ok(x) => {
            if x > 0.0 {
                Ok(x)
            } else {
                Err(format!(
                    "capacity_multiplier must be higher than 0. {x} received."
                ))
            }
        }
        Err(e) => Err(e.to_string()),
    }
}

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Path to a directory containing simulation files, and where simulation results will be stored
    #[clap(long, short, default_value = DEFAULT_DATA_DIR)]
    data_dir: PathBuf,
    /// Path to the simulation file to be used by the simulator
    /// This can either be an absolute path, or relative path with respect to data_dir
    #[clap(long, short, default_value = DEFAULT_SIM_FILE)]
    sim_file: PathBuf,
    /// Total time the simulator will be running
    #[clap(long, short)]
    total_time: Option<u32>,
    /// Number of activity results to batch together before printing to csv file [min: 1]
    #[clap(long, short, default_value_t = DEFAULT_PRINT_BATCH_SIZE, value_parser = clap::builder::RangedU64ValueParser::<u32>::new().range(1..u32::MAX as u64))]
    print_batch_size: u32,
    /// Level of verbosity of the messages displayed by the simulator.
    /// Possible values: [off, error, warn, info, debug, trace]
    #[clap(long, short, verbatim_doc_comment, default_value = "info")]
    log_level: LevelFilter,
    /// Expected payment amount for the random activity generator
    #[clap(long, short, default_value_t = EXPECTED_PAYMENT_AMOUNT, value_parser = clap::builder::RangedU64ValueParser::<u64>::new().range(1..u64::MAX))]
    expected_pmt_amt: u64,
    /// Multiplier of the overall network capacity used by the random activity generator
    #[clap(long, short, default_value_t = ACTIVITY_MULTIPLIER, value_parser = clap::builder::StringValueParser::new().try_map(deserialize_f64_greater_than_zero))]
    capacity_multiplier: f64,
    /// Do not create an output file containing the simulations results
    #[clap(long, default_value_t = false)]
    no_results: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    SimpleLogger::new()
        .with_level(LevelFilter::Warn)
        .with_module_level("sim_lib", cli.log_level)
        .with_module_level("sim_cli", cli.log_level)
        .init()
        .unwrap();

    let sim_path = read_sim_path(cli.data_dir.clone(), cli.sim_file).await?;
    let SimParams { nodes, activity } =
        serde_json::from_str(&std::fs::read_to_string(sim_path)?)
            .map_err(|e| anyhow!("Could not deserialize node connection data or activity description from simulation file (line {}, col {}).", e.line(), e.column()))?;

    let mut clients: HashMap<PublicKey, Arc<Mutex<dyn LightningNode>>> = HashMap::new();
    let mut pk_node_map = HashMap::new();
    let mut alias_node_map = HashMap::new();

    for connection in nodes {
        // TODO: Feels like there should be a better way of doing this without having to Arc<Mutex<T>>> it at this time.
        // Box sort of works, but we won't know the size of the dyn LightningNode at compile time so the compiler will
        // scream at us when trying to create the Arc<Mutex>> later on while adding the node to the clients map
        let node: Arc<Mutex<dyn LightningNode>> = match connection {
            NodeConnection::LND(c) => Arc::new(Mutex::new(LndNode::new(c).await?)),
            NodeConnection::CLN(c) => Arc::new(Mutex::new(ClnNode::new(c).await?)),
        };

        let node_info = node.lock().await.get_info().clone();

        log::info!(
            "Connected to {} - Node ID: {}.",
            node_info.alias,
            node_info.pubkey
        );

        if clients.contains_key(&node_info.pubkey) {
            anyhow::bail!(LightningError::ValidationError(format!(
                "duplicated node: {}.",
                node_info.pubkey
            )));
        }

        if alias_node_map.contains_key(&node_info.alias) {
            anyhow::bail!(LightningError::ValidationError(format!(
                "duplicated node: {}.",
                node_info.alias
            )));
        }

        clients.insert(node_info.pubkey, node);
        pk_node_map.insert(node_info.pubkey, node_info.clone());
        alias_node_map.insert(node_info.alias.clone(), node_info);
    }

    let mut validated_activities = vec![];
    // Make all the activities identifiable by PK internally
    for act in activity.into_iter() {
        // We can only map aliases to nodes we control, so if either the source or destination alias
        // is not in alias_node_map, we fail
        let source = if let Some(source) = match &act.source {
            NodeId::PublicKey(pk) => pk_node_map.get(pk),
            NodeId::Alias(a) => alias_node_map.get(a),
        } {
            source.clone()
        } else {
            anyhow::bail!(LightningError::ValidationError(format!(
                "activity source {} not found in nodes.",
                act.source
            )));
        };

        let destination = match &act.destination {
            NodeId::Alias(a) => {
                if let Some(info) = alias_node_map.get(a) {
                    info.clone()
                } else {
                    anyhow::bail!(LightningError::ValidationError(format!(
                        "unknown activity destination: {}.",
                        act.destination
                    )));
                }
            }
            NodeId::PublicKey(pk) => {
                if let Some(info) = pk_node_map.get(pk) {
                    info.clone()
                } else {
                    clients
                        .get(&source.pubkey)
                        .unwrap()
                        .lock()
                        .await
                        .get_node_info(pk)
                        .await
                        .map_err(|e| {
                            log::debug!("{}", e);
                            LightningError::ValidationError(format!(
                                "Destination node unknown or invalid: {}.",
                                pk,
                            ))
                        })?
                }
            }
        };

        validated_activities.push(ActivityDefinition {
            source,
            destination,
            interval_secs: act.interval_secs,
            amount_msat: act.amount_msat,
        });
    }
    let write_results = if !cli.no_results {
        Some(WriteResults {
            results_dir: mkdir(cli.data_dir.join("results")).await?,
            batch_size: cli.print_batch_size,
        })
    } else {
        None
    };

    let (shutdown_trigger, shutdown_listener) = triggered::trigger();

    let channels = generate_sim_nodes();
    let graph = match SimGraph::new(channels.clone(), shutdown_trigger.clone()) {
        Ok(graph) => Arc::new(Mutex::new(graph)),
        Err(e) => anyhow::bail!("failed: {:?}", e),
    };

    let routing_graph = match populate_network_graph(channels) {
        Ok(r) => r,
        Err(e) => anyhow::bail!("failed: {:?}", e),
    };

    let sim = Simulation::new(
        ln_node_from_graph(graph.clone(), Arc::new(routing_graph)).await,
        validated_activities,
        cli.total_time,
        cli.expected_pmt_amt,
        cli.capacity_multiplier,
        write_results,
        (shutdown_trigger, shutdown_listener),
    );
    let sim2 = sim.clone();

    ctrlc::set_handler(move || {
        log::info!("Shutting down simulation.");
        sim2.shutdown();
    })?;

    // Run the simulation (blocking) until it exits. Once this happens, we can also wait for the simulated graph to
    // shut down. Errors in either of these will universally trigger shutdown because we share a trigger, so it doesn't
    // matter what order we wait for these.
    sim.run().await?;
    graph.lock().await.wait_for_shutdown().await;

    Ok(())
}

fn generate_sim_nodes() -> Vec<SimulatedChannel> {
    let capacity = 300000000;
    let mut channels: Vec<SimulatedChannel> = vec![];
    let (_, first_node) = get_random_keypair();

    // Create channels in a ring so that we'll get long payment paths.
    let mut node_1 = first_node;
    for i in 0..10 {
        // Create a new node that we'll create a channel with. If we're on the last node in the circle, we'll loop
        // back around to the first node to close it.
        let node_2 = if i == 10 {
            first_node
        } else {
            let (_, pk) = get_random_keypair();
            pk
        };

        let node_1_to_2 = ChannelPolicy {
            pubkey: node_1,
            max_htlc_count: 483,
            max_in_flight_msat: capacity / 2,
            min_htlc_size_msat: 1,
            max_htlc_size_msat: capacity / 2,
            cltv_expiry_delta: 40,
            // Alter fee rate a little for different values.
            base_fee: 1000 * i,
            fee_rate_prop: 1500 * i,
        };

        let node_2_to_1 = ChannelPolicy {
            pubkey: node_2,
            max_htlc_count: 483,
            max_in_flight_msat: capacity / 2,
            min_htlc_size_msat: 1,
            max_htlc_size_msat: capacity / 2,
            cltv_expiry_delta: 40 + 10 * i as u32,
            // Alter fee rate a little for different values.
            base_fee: 2000 * i,
            fee_rate_prop: i,
        };

        channels.push(SimulatedChannel::new(
            capacity,
            // Unique channel ID per link.
            100 + i,
            node_1_to_2,
            node_2_to_1,
        ));

        // Once we've created this link in the circle, progress our current node to be node_1 so that we can generate
        // a new edge.
        node_1 = node_2;
    }

    channels
}

/// COPIED from test utils!
pub fn get_random_bytes(size: usize) -> Vec<u8> {
    rand::thread_rng()
        .sample_iter(Uniform::new(u8::MIN, u8::MAX))
        .take(size)
        .collect()
}

pub fn get_random_int(s: u64, e: u64) -> u64 {
    rand::thread_rng().gen_range(s..e)
}

pub fn get_random_keypair() -> (SecretKey, PublicKey) {
    loop {
        if let Ok(sk) = SecretKey::from_slice(&get_random_bytes(32)) {
            return (sk, PublicKey::from_secret_key(&Secp256k1::new(), &sk));
        }
    }
}
/// COPIED from test utils!

async fn read_sim_path(data_dir: PathBuf, sim_file: PathBuf) -> anyhow::Result<PathBuf> {
    let sim_path = if sim_file.is_relative() {
        data_dir.join(sim_file)
    } else {
        sim_file
    };

    if sim_path.exists() {
        Ok(sim_path)
    } else {
        log::info!("Simulation file '{}' does not exist.", sim_path.display());
        select_sim_file(data_dir).await
    }
}

async fn select_sim_file(data_dir: PathBuf) -> anyhow::Result<PathBuf> {
    let sim_files = std::fs::read_dir(data_dir.clone())?
        .filter_map(|f| {
            f.ok().and_then(|f| {
                if f.path().extension()?.to_str()? == "json" {
                    f.file_name().into_string().ok()
                } else {
                    None
                }
            })
        })
        .collect::<Vec<_>>();

    if sim_files.is_empty() {
        anyhow::bail!(
            "no simulation files found in {}.",
            data_dir.canonicalize()?.display()
        );
    }

    let selection = dialoguer::Select::new()
        .with_prompt(format!(
            "Select a simulation file. Found these in {}",
            data_dir.canonicalize()?.display()
        ))
        .items(&sim_files)
        .default(0)
        .interact()?;

    Ok(data_dir.join(sim_files[selection].clone()))
}

async fn mkdir(dir: PathBuf) -> anyhow::Result<PathBuf> {
    tokio::fs::create_dir_all(&dir).await?;
    Ok(dir)
}
