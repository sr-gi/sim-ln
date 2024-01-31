use crate::{LightningError, LightningNode, NodeInfo, PaymentOutcome, PaymentResult};
use async_trait::async_trait;
use bitcoin::constants::ChainHash;
use bitcoin::hashes::{sha256::Hash as Sha256, Hash};
use bitcoin::secp256k1::PublicKey;
use bitcoin::{Network, ScriptBuf, TxOut};
use lightning::ln::chan_utils::make_funding_redeemscript;
use std::collections::{hash_map::Entry, HashMap};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use lightning::ln::features::{ChannelFeatures, NodeFeatures};
use lightning::ln::msgs::{
    LightningError as LdkError, UnsignedChannelAnnouncement, UnsignedChannelUpdate,
};
use lightning::ln::{PaymentHash, PaymentPreimage};
use lightning::routing::gossip::{NetworkGraph, NodeId};
use lightning::routing::router::{
    find_route, Path, Payee, PaymentParameters, Route, RouteParameters,
};
use lightning::routing::scoring::{
    ProbabilisticScorer, ProbabilisticScoringDecayParameters, ProbabilisticScoringFeeParameters,
};
use lightning::routing::utxo::{UtxoLookup, UtxoResult};
use lightning::util::logger::{Level, Logger, Record};
use thiserror::Error;
use tokio::select;
use tokio::sync::oneshot::{channel, Receiver, Sender};
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use triggered::{Listener, Trigger};

/// ForwardingError represents the various errors that we can run into when forwarding payments in a simulated network.
/// Since we're not using real lightning nodes, these errors are not obfuscated and can be propagated to the sending
/// node and used for analysis.
#[derive(Debug, Error)]
pub enum ForwardingError {
    /// Zero amount htlcs are in valid in the protocol.
    #[error("ZeroAmountHtlc")]
    ZeroAmountHtlc,
    /// The outgoing channel id was not found in the network graph.
    #[error("ChannelNotFound: {0}")]
    ChannelNotFound(u64),
    /// The node pubkey provided was not associated with the channel in the network graph.
    #[error("NodeNotFound: {0:?}")]
    NodeNotFound(PublicKey),
    /// The channel has already forwarded a HTLC with the payment hash provided (to be removed when MPP support is
    /// added).
    #[error("PaymentHashExists: {0:?}")]
    PaymentHashExists(PaymentHash),
    /// A htlc with the payment hash provided could not be found to resolve.
    #[error("PaymentHashNotFound: {0:?}")]
    PaymentHashNotFound(PaymentHash),
    /// The forwarding node did not have sufficient outgoing balance to forward the htlc (htlc amount / balance).
    #[error("InsufficientBalance: amount: {0} > balance: {1}")]
    InsufficientBalance(u64, u64),
    /// The htlc forwarded is less than the channel's advertised minimum htlc amount (htlc amount / minimum).
    #[error("LessThanMinimum: amount: {0} < minimum: {1}")]
    LessThanMinimum(u64, u64),
    /// The htlc forwarded is more than the channel's advertised maximum htlc amount (htlc amount / maximum).
    #[error("MoreThanMaximum: amount: {0} > maximum: {1}")]
    MoreThanMaximum(u64, u64),
    /// The channel has reached its maximum allowable number of htlcs in flight (total in flight / maximim).
    #[error("ExceedsInFlightCount: total in flight: {0} > maximum count: {1}")]
    ExceedsInFlightCount(u64, u64),
    /// The forwarded htlc's amount would push the channel over its maximum allowable in flight total
    /// (total in flight / maximum).
    #[error("ExceedsInFlightTotal: total in flight amount: {0} > maximum amount: {0}")]
    ExceedsInFlightTotal(u64, u64),
    /// The forwarded htlc's cltv expiry exceeds the maximum value used to express block heights in bitcoin.
    #[error("ExpiryInSeconds: cltv expressed in seconds: {0}")]
    ExpiryInSeconds(u32),
    /// The forwarded htlc has insufficient cltv delta for the channel's minimum delta (cltv delta / minimum).
    #[error("InsufficientCltvDelta: cltv delta: {0} < required: {1}")]
    InsufficientCltvDelta(u32, u32),
    /// The forwarded htlc has insufficient fee for the channel's policy (fee / base fee / prop fee / expected fee).
    #[error("InsufficientFee: offered fee: {0} (base: {1}, prop: {2}) < expected: {3}")]
    InsufficientFee(u64, u64, u64, u64),
    #[error("SanityCheckFailed: capacity: {0} != node_1: {1} + node_2: {2}")]
    /// Sanity check on channel balances failed (capacity / node_1 balance / node_2 balance).
    SanityCheckFailed(u64, u64, u64),
}

impl ForwardingError {
    /// Returns a boolean indicating whether failure to forward a htlc is a critical error that warrants shutdown.
    fn is_critical(&self) -> bool {
        matches!(
            self,
            ForwardingError::ZeroAmountHtlc
                | ForwardingError::ChannelNotFound(_)
                | ForwardingError::NodeNotFound(_)
                | ForwardingError::PaymentHashExists(_)
                | ForwardingError::PaymentHashNotFound(_)
                | ForwardingError::SanityCheckFailed(_, _, _)
        )
    }
}

/// Represents an in-flight htlc that has been forwarded over a channel that is awaiting resolution.
#[derive(Copy, Clone)]
struct Htlc {
    hash: PaymentHash,
    amount_msat: u64,
    cltv_expiry: u32,
}

/// Represents one node in the channel's forwarding policy and restrictions. Note that this doesn't directly map to
/// a single concept in the protocol, a few things have been combined for the sake of simplicity. Used to manage the
/// lightning "state machine" and check that HTLCs are added in accordance of the advertised policy.
#[derive(Clone)]
pub struct ChannelPolicy {
    pub pubkey: PublicKey,
    pub max_htlc_count: u64,
    pub max_in_flight_msat: u64,
    pub min_htlc_size_msat: u64,
    pub max_htlc_size_msat: u64,
    pub cltv_expiry_delta: u32,
    pub base_fee: u64,
    pub fee_rate_prop: u64,
}

/// The internal state of one side of a simulated channel, including its forwarding parameters. This struct is
/// primarily responsible for handling our view of what's currently in-flight on the channel, and how much
/// liquidity we have.
#[derive(Clone)]
struct ChannelState {
    local_balance_msat: u64,
    in_flight: HashMap<PaymentHash, Htlc>,
    policy: ChannelPolicy,
}

impl ChannelState {
    /// Creates a new channel with local liquidity as allocated by the caller. The responsibility of ensuring that the
    /// local balance of each side of the channel equals its total capacity is on the caller, as we are only dealing
    /// with a one-sided view of the channel's state.
    fn new(policy: ChannelPolicy, local_balance_msat: u64) -> Self {
        ChannelState {
            local_balance_msat,
            in_flight: HashMap::new(),
            policy,
        }
    }

    /// Returns the sum of all the *in flight outgoing* HTLCs on the channel.
    fn in_flight_total(&self) -> u64 {
        self.in_flight
            .iter()
            .fold(0, |sum, val| sum + val.1.amount_msat)
    }

    /// Checks whether the proposed HTLC abides by the channel policy advertised for using this channel as the
    /// *outgoing* link in a forward.
    fn check_htlc_forward(
        &self,
        cltv_delta: u32,
        amt: u64,
        fee: u64,
    ) -> Result<(), ForwardingError> {
        if cltv_delta < self.policy.cltv_expiry_delta {
            return Err(ForwardingError::InsufficientCltvDelta(
                cltv_delta,
                self.policy.cltv_expiry_delta,
            ));
        }

        // As u64 will round expected fee down to nearest msat (this is what the protocol dictates).
        let expected_fee = (self.policy.base_fee as f64
            + ((self.policy.fee_rate_prop as f64 * amt as f64) / 1000000.0))
            as u64;
        if fee < expected_fee {
            return Err(ForwardingError::InsufficientFee(
                fee,
                self.policy.base_fee,
                self.policy.fee_rate_prop,
                expected_fee,
            ));
        }

        Ok(())
    }

    /// Checks whether the proposed HTLC can be added to the channel as an outgoing HTLC. This requires that we have
    /// sufficient liquidity, and that the restrictions on our in flight htlc balance and count are not violated by
    /// the addition of the HTLC. Specification sanity checks (such as reasonable CLTV) are also included, as this
    /// is where we'd check it in real life.
    fn check_outgoing_addition(&self, htlc: &Htlc) -> Result<(), ForwardingError> {
        // Fails if the value provided fails its inequality check without policy.
        macro_rules! fail_policy_inequality {
			($value:expr, $op:tt, $field:ident, $error_variant:ident) => {
				if $value $op self.policy.$field {
					return Err(ForwardingError::$error_variant(
							$value,
							self.policy.$field,
							));
				}
			};
		}

        fail_policy_inequality!(htlc.amount_msat, >, max_htlc_size_msat, MoreThanMaximum);
        fail_policy_inequality!(htlc.amount_msat, <, min_htlc_size_msat, LessThanMinimum);
        fail_policy_inequality!(self.in_flight.len() as u64 + 1, >, max_htlc_count, ExceedsInFlightCount);
        fail_policy_inequality!(
            self.in_flight_total() + htlc.amount_msat, >, max_in_flight_msat, ExceedsInFlightTotal
        );

        // Values that are not on self.policy don't use the macro.
        if htlc.amount_msat > self.local_balance_msat {
            return Err(ForwardingError::InsufficientBalance(
                htlc.amount_msat,
                self.local_balance_msat,
            ));
        }

        if htlc.cltv_expiry > 500000000 {
            return Err(ForwardingError::ExpiryInSeconds(htlc.cltv_expiry));
        }

        Ok(())
    }

    /// Adds the HTLC to our set of outgoing in-flight HTLCs. [`check_outgoing_addition`] should be called before
    /// this to ensure that the restrictions on outgoing HTLCs are not violated. Local balance is decreased by the
    /// HTLC amount, as this liquidity is no longer available.
    ///
    /// Note: MPP payments are not currently supported, so this function will fail if a duplicate payment hash is
    /// reported.
    fn add_outgoing_htlc(&mut self, htlc: Htlc) -> Result<(), ForwardingError> {
        self.check_outgoing_addition(&htlc)?;

        match self.in_flight.get(&htlc.hash) {
            Some(_) => Err(ForwardingError::PaymentHashExists(htlc.hash)),
            None => {
                self.local_balance_msat -= htlc.amount_msat;
                self.in_flight.insert(htlc.hash, htlc);
                Ok(())
            }
        }
    }

    /// Removes the HTLC from our set of outgoing in-flight HTLCs, failing if the payment hash is not found. If the
    /// HTLC failed, the balance is returned to our local liquidity. Note that this function is not responsible for
    /// reflecting that the balance has moved to the other side of the channel in the success-case, calling code is
    /// responsible for that.
    fn remove_outgoing_htlc(
        &mut self,
        hash: PaymentHash,
        success: bool,
    ) -> Result<Htlc, ForwardingError> {
        match self.in_flight.remove(&hash) {
            Some(v) => {
                // If the HTLC failed, pending balance returns to local balance.
                if !success {
                    self.local_balance_msat += v.amount_msat
                }

                Ok(v)
            }
            None => Err(ForwardingError::PaymentHashNotFound(hash)),
        }
    }
}

/// Represents a simulated channel, and is responsible for managing addition and removal of HTLCs from the channel and
/// sanity checks. Channel state is tracked *unidirectionally* for each participant in the channel.
///
/// Each node represented in the channel tracks only its outgoing HTLCs, and balance is transferred between the two
/// nodes as they settle or fail. Given some channel: node_1 <----> node_2:
/// * HTLC sent node_1 -> node_2: added to in-flight outgoing htlcs on node_1.
/// * HTLC sent node_2 -> node_1: added to in-flight outgoing htlcs on node_1.
///
/// Rules for managing balance are as follows:
/// * When a HTLC is in flight, the channel's local outgoing liquidity decreases (as it's locked up).
/// * When a HTLC fails, the balance is returned to the local node (the one that it was in-flight / outgoing on).
/// * When a HTLC succeeds, the balance is sent to the remote node (the one that did not track it as in-flight).
///
/// With each state transition, the simulated channel checks that the sum of its local balances and in-flight equal the
/// total channel capacity. Failure of this sanity check represents a critical failure in the state machine.
#[derive(Clone)]
pub struct SimulatedChannel {
    capacity_msat: u64,
    short_channel_id: u64,
    node_1: ChannelState,
    node_2: ChannelState,
}

impl SimulatedChannel {
    /// Creates a new channel with the capacity and policies provided. The total capacity of the channel is evenly split
    /// between the channel participants (this is an arbitrary decision).
    pub fn new(
        capacity_msat: u64,
        short_channel_id: u64,
        node_1: ChannelPolicy,
        node_2: ChannelPolicy,
    ) -> Self {
        SimulatedChannel {
            capacity_msat,
            short_channel_id,
            node_1: ChannelState::new(node_1, capacity_msat / 2),
            node_2: ChannelState::new(node_2, capacity_msat / 2),
        }
    }

    /// Adds a htlc to the appropriate side of the simulated channel, checking its policy and balance are okay.
    fn add_htlc(&mut self, node: PublicKey, htlc: Htlc) -> Result<(), ForwardingError> {
        if htlc.amount_msat == 0 {
            return Err(ForwardingError::ZeroAmountHtlc);
        }

        if node == self.node_1.policy.pubkey {
            self.node_1.add_outgoing_htlc(htlc)?;
            return self.sanity_check();
        }

        if node == self.node_2.policy.pubkey {
            self.node_2.add_outgoing_htlc(htlc)?;
            return self.sanity_check();
        }

        Err(ForwardingError::NodeNotFound(node))
    }

    /// Performs a sanity check on the total balances in a channel. Note that we do not currently include on-chain
    /// fees or reserve so these values should exactly match.
    fn sanity_check(&self) -> Result<(), ForwardingError> {
        let node_1_total = self.node_1.local_balance_msat + self.node_1.in_flight_total();
        let node_2_total = self.node_2.local_balance_msat + self.node_2.in_flight_total();

        if node_1_total + node_2_total != self.capacity_msat {
            return Err(ForwardingError::SanityCheckFailed(
                self.capacity_msat,
                node_1_total,
                node_2_total,
            ));
        }

        Ok(())
    }

    /// Removes a htlc from the appropriate size of the simulated channel, settling balances across channel sides
    /// based on the success of the htlc.
    fn remove_htlc(
        &mut self,
        incoming_node: PublicKey,
        hash: PaymentHash,
        success: bool,
    ) -> Result<(), ForwardingError> {
        // Removes the HTLC from the node that it was added to as an outgoing HTLC. If it succeeded, move the balance
        // over to the other side of the channel. The HTLC removal will handle returning balance to the local channel
        // in the case of a failure.
        macro_rules! process_outgoing_htlc {
			($self:ident, $sender:ident, $receiver:ident, $hash:expr, $success:expr) => {
				match $self.$sender.remove_outgoing_htlc($hash, $success){
					// If the HTLC was settled, its amount is transferred to the remote party's local balance.
					// If it was failed, the above removal has already dealt with balance management.
					Ok(htlc) => {
						if $success {
							$self.$receiver.local_balance_msat += htlc.amount_msat;
						}

						return $self.sanity_check();
					},
					Err(e) => Err(e),
				}
			};
		}

        if incoming_node == self.node_1.policy.pubkey {
            return process_outgoing_htlc!(self, node_1, node_2, hash, success);
        }

        if incoming_node == self.node_2.policy.pubkey {
            return process_outgoing_htlc!(self, node_2, node_1, hash, success);
        }

        Err(ForwardingError::NodeNotFound(incoming_node))
    }

    /// Checks a htlc forward against the outgoing policy of the node provided.
    fn check_htlc_forward(
        &self,
        node: PublicKey,
        cltv_delta: u32,
        amount_msat: u64,
        fee_msat: u64,
    ) -> Result<(), ForwardingError> {
        if node == self.node_1.policy.pubkey {
            return self
                .node_1
                .check_htlc_forward(cltv_delta, amount_msat, fee_msat);
        }

        if node == self.node_2.policy.pubkey {
            return self
                .node_2
                .check_htlc_forward(cltv_delta, amount_msat, fee_msat);
        }

        Err(ForwardingError::NodeNotFound(node))
    }
}

/// SimNetwork represents a high level network coordinator that is responsible for the task of actually propagating
/// payments through the simulated network.
#[async_trait]
trait SimNetwork: Send + Sync {
    /// Sends payments over the route provided through the network, reporting the final payment outcome to the sender
    /// channel provided.
    fn dispatch_payment(
        &mut self,
        source: PublicKey,
        route: Route,
        preimage: PaymentPreimage,
        sender: Sender<Result<PaymentResult, LightningError>>,
    );

    /// Looks up a node in the simulated network and a list of its channel capacities.
    async fn lookup_node(&self, node: &PublicKey) -> Result<(NodeInfo, Vec<u64>), LightningError>;
}

/// A wrapper struct used to implement the LightningNode trait (can be thought of as "the" lightning node). Passes
/// all functionality through to a coordinating simulation network. This implementation contains both the [`SimNetwork`]
/// implementation that will allow us to dispatch payments and a read-only NetworkGraph that is used for pathfinding.
/// While these two could be combined, we re-use the LDK-native struct to allow re-use of their pathfinding logic.
struct SimNode<'a, T: SimNetwork> {
    info: NodeInfo,
    /// The underlying execution network that will be responsible for dispatching payments.
    network: Arc<Mutex<T>>,
    /// Tracks the channel that will provide updates for payments by hash.
    in_flight: HashMap<PaymentHash, Receiver<Result<PaymentResult, LightningError>>>,
    /// A read-only graph used for pathfinding.
    pathfinding_graph: Arc<NetworkGraph<&'a WrappedLog>>,
}

impl<'a, T: SimNetwork> SimNode<'a, T> {
    /// Creates a new simulation node that refers to the high level network coordinator provided to process payments
    /// on its behalf. The pathfinding graph is provided separately so that each node can handle its own pathfinding.
    pub fn new(
        pubkey: PublicKey,
        payment_network: Arc<Mutex<T>>,
        pathfinding_graph: Arc<NetworkGraph<&'a WrappedLog>>,
    ) -> Self {
        SimNode {
            info: node_info(pubkey),
            network: payment_network,
            in_flight: HashMap::new(),
            pathfinding_graph,
        }
    }
}

/// Produces the node info for a mocked node, filling in the features that the simulator requires.
fn node_info(pk: PublicKey) -> NodeInfo {
    // Set any features that the simulator requires here.
    let mut features = NodeFeatures::empty();
    features.set_keysend_optional();

    NodeInfo {
        pubkey: pk,
        alias: "".to_string(), // TODO: store alias?
        features,
    }
}

/// Uses LDK's pathfinding algorithm with default parameters to find a path from source to destination, with no
/// restrictions on fee budget.
fn find_payment_route(
    source: PublicKey,
    dest: PublicKey,
    amount_msat: u64,
    pathfinding_graph: &NetworkGraph<&WrappedLog>,
) -> Result<Route, LdkError> {
    let params = ProbabilisticScoringDecayParameters::default();
    let scorer = ProbabilisticScorer::new(params, pathfinding_graph, &WrappedLog {});

    find_route(
        &source,
        &RouteParameters {
            payment_params: PaymentParameters {
                payee: Payee::Clear {
                    node_id: dest,
                    route_hints: Vec::new(),
                    features: None,
                    // We don't currently bother with final CLTV delta.
                    final_cltv_expiry_delta: 0,
                },
                expiry_time: None,
                max_total_cltv_expiry_delta: u32::MAX,
                // TODO: set non-zero value to support MPP.
                max_path_count: 1,
                // Allow sending htlcs up to 50% of the channel's capacity.
                max_channel_saturation_power_of_half: 1,
                previously_failed_channels: Vec::new(),
                previously_failed_blinded_path_idxs: Vec::new(),
            },
            final_value_msat: amount_msat,
            max_total_routing_fee_msat: None,
        },
        pathfinding_graph,
        None,
        &WrappedLog {},
        &scorer,
        &ProbabilisticScoringFeeParameters::default(),
        &[0; 32],
    )
}

#[async_trait]
impl<T: SimNetwork> LightningNode for SimNode<'_, T> {
    fn get_info(&self) -> &NodeInfo {
        &self.info
    }

    async fn get_network(&mut self) -> Result<Network, LightningError> {
        Ok(Network::Regtest)
    }

    /// send_payment picks a random preimage for a payment, dispatches it in the network and adds a tracking channel
    /// to our node state to be used for subsequent track_payment calls.
    async fn send_payment(
        &mut self,
        dest: PublicKey,
        amount_msat: u64,
    ) -> Result<PaymentHash, LightningError> {
        // Create a sender and receiver pair that will be used to report the results of the payment and add them to
        // our internal tracking state along with the chosen payment hash.
        let (sender, receiver) = channel();
        let preimage = PaymentPreimage(rand::random());
        let preimage_bytes = Sha256::hash(&preimage.0[..]).to_byte_array();
        let payment_hash = PaymentHash(preimage_bytes);

        self.in_flight.insert(payment_hash, receiver);

        let route = match find_payment_route(
            self.info.pubkey,
            dest,
            amount_msat,
            &self.pathfinding_graph,
        ) {
            Ok(path) => path,
            // In the case that we can't find a route for the payment, we still report a successful payment *api call*
            // and report RouteNotFound to the tracking channel. This mimics the behavior of real nodes.
            Err(e) => {
                log::trace!("Could not find path for payment: {:?}.", e);

                if let Err(e) = sender.send(Ok(PaymentResult {
                    htlc_count: 0,
                    payment_outcome: PaymentOutcome::RouteNotFound,
                })) {
                    log::error!("Could not send payment result: {:?}.", e);
                }

                return Ok(payment_hash);
            }
        };

        // If we did successfully obtain a route, dispatch the payment through the network and then report success.
        self.network
            .lock()
            .await
            .dispatch_payment(self.info.pubkey, route, preimage, sender);

        Ok(payment_hash)
    }

    /// track_payment blocks until a payment outcome is returned for the payment hash provided, or the shutdown listener
    /// provided is triggered. This call will fail if the hash provided was not obtained by calling send_payment first.
    async fn track_payment(
        &mut self,
        hash: PaymentHash,
        listener: Listener,
    ) -> Result<PaymentResult, LightningError> {
        match self.in_flight.get_mut(&hash) {
            Some(receiver) => {
                select! {
                    biased;
                    _ = listener => Err(LightningError::TrackPaymentError("shutdown during payment tracking".to_string())),

                    // If we get a payment result back, remove from our in flight set of payments and return the result.
                    res = receiver => {
                        self.in_flight.remove(&hash);
                        res.map_err(|e| LightningError::TrackPaymentError(format!("channel receive err: {}", e)))?
                    },
                }
            }
            None => Err(LightningError::TrackPaymentError(format!(
                "payment hash {} not found",
                hex::encode(hash.0),
            ))),
        }
    }

    async fn get_node_info(&mut self, node_id: &PublicKey) -> Result<NodeInfo, LightningError> {
        Ok(self.network.lock().await.lookup_node(node_id).await?.0)
    }

    async fn list_channels(&mut self) -> Result<Vec<u64>, LightningError> {
        Ok(self
            .network
            .lock()
            .await
            .lookup_node(&self.info.pubkey)
            .await?
            .1)
    }
}

/// Graph is the top level struct that is used to coordinate simulation of lightning nodes.
pub struct SimGraph {
    /// nodes caches the list of nodes in the network with a vector of their channel capacities, only used for quick
    /// lookup.
    nodes: HashMap<PublicKey, Vec<u64>>,

    /// channels maps the scid of a channel to its current simulation state.
    channels: Arc<Mutex<HashMap<u64, SimulatedChannel>>>,

    /// track all tasks spawned to process payments in the graph.
    tasks: JoinSet<()>,

    /// trigger shutdown if a critical error occurs.
    shutdown_trigger: Trigger,
}

impl SimGraph {
    /// Creates a graph on which to simulate payments.
    pub fn new(
        graph_channels: Vec<SimulatedChannel>,
        shutdown_trigger: Trigger,
    ) -> Result<Self, LdkError> {
        let mut nodes: HashMap<PublicKey, Vec<u64>> = HashMap::new();
        let mut channels = HashMap::new();

        for channel in graph_channels.iter() {
            channels.insert(channel.short_channel_id, channel.clone());

            macro_rules! insert_node_entry {
                ($pubkey:expr) => {{
                    match nodes.entry($pubkey) {
                        Entry::Occupied(o) => o.into_mut().push(channel.capacity_msat),
                        Entry::Vacant(v) => {
                            v.insert(vec![channel.capacity_msat]);
                        }
                    }
                }};
            }

            insert_node_entry!(channel.node_1.policy.pubkey);
            insert_node_entry!(channel.node_2.policy.pubkey);
        }

        Ok(SimGraph {
            nodes,
            channels: Arc::new(Mutex::new(channels)),
            tasks: JoinSet::new(),
            shutdown_trigger,
        })
    }

    /// Blocks until all tasks created by the simulator have shut down. This function does not trigger shutdown,
    /// because it expects erroring-out tasks to handle their own shutdown triggering.
    pub async fn wait_for_shutdown(&mut self) {
        log::debug!("Waiting for simulated graph to shutdown.");

        while let Some(res) = self.tasks.join_next().await {
            if let Err(e) = res {
                log::error!("Graph task exited with error: {e}");
            }
        }

        log::debug!("Simulated graph shutdown.");
    }
}

/// Produces a map of node public key to lightning node implementation to be used for simulations.
pub async fn ln_node_from_graph<'a>(
    graph: Arc<Mutex<SimGraph>>,
    routing_graph: Arc<NetworkGraph<&'_ WrappedLog>>,
) -> HashMap<PublicKey, Arc<Mutex<dyn LightningNode + '_>>> {
    let mut nodes: HashMap<PublicKey, Arc<Mutex<dyn LightningNode>>> = HashMap::new();

    for pk in graph.lock().await.nodes.keys() {
        nodes.insert(
            *pk,
            Arc::new(Mutex::new(SimNode::new(
                *pk,
                graph.clone(),
                routing_graph.clone(),
            ))),
        );
    }

    nodes
}

/// Populates a network graph based on the set of simulated channels provided. This function *only* applies channel
/// announcements, which has the effect of adding the nodes in each channel to the graph, because LDK does not export
/// all of the fields required to apply node announcements. This means that we will not have node-level information
/// (such as features) available in the routing graph.
pub fn populate_network_graph<'a>(
    channels: Vec<SimulatedChannel>,
) -> Result<NetworkGraph<&'a WrappedLog>, LdkError> {
    let graph = NetworkGraph::new(Network::Regtest, &WrappedLog {});

    let chain_hash = ChainHash::using_genesis_block(Network::Regtest);

    for channel in channels {
        let announcement = UnsignedChannelAnnouncement {
            // For our purposes we don't currently need any channel level features.
            features: ChannelFeatures::empty(),
            chain_hash,
            short_channel_id: channel.short_channel_id,
            node_id_1: NodeId::from_pubkey(&channel.node_1.policy.pubkey),
            node_id_2: NodeId::from_pubkey(&channel.node_2.policy.pubkey),
            // Note: we don't need bitcoin keys for our purposes, so we just copy them *but* remember that we do use
            // this for our fake utxo validation so they do matter for producing the script that we mock validate.
            bitcoin_key_1: NodeId::from_pubkey(&channel.node_1.policy.pubkey),
            bitcoin_key_2: NodeId::from_pubkey(&channel.node_2.policy.pubkey),
            // Internal field used by LDK, we don't need it.
            excess_data: Vec::new(),
        };

        let utxo_validator = UtxoValidator {
            amount_sat: channel.capacity_msat / 1000,
            script: make_funding_redeemscript(
                &channel.node_1.policy.pubkey,
                &channel.node_2.policy.pubkey,
            )
            .to_v0_p2wsh(),
        };

        graph.update_channel_from_unsigned_announcement(&announcement, &Some(&utxo_validator))?;

        macro_rules! generate_and_update_channel {
            ($node:expr, $flags:expr) => {{
                let update = UnsignedChannelUpdate {
                    chain_hash,
                    short_channel_id: channel.short_channel_id,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs() as u32,
                    flags: $flags,
                    cltv_expiry_delta: $node.policy.cltv_expiry_delta as u16,
                    htlc_minimum_msat: $node.policy.min_htlc_size_msat,
                    htlc_maximum_msat: $node.policy.max_htlc_size_msat,
                    fee_base_msat: $node.policy.base_fee as u32,
                    fee_proportional_millionths: $node.policy.fee_rate_prop as u32,
                    excess_data: Vec::new(),
                };

                graph.update_channel_unsigned(&update)?;
            }};
        }

        // The least significant bit of the channel flag field represents the direction that the channel update
        // applies to. This value is interpreted as node_1 if it is zero, and node_2 otherwise.
        generate_and_update_channel!(channel.node_1, 0);
        generate_and_update_channel!(channel.node_2, 1);
    }

    Ok(graph)
}

#[async_trait]
impl SimNetwork for SimGraph {
    /// dispatch_payment asynchronously propagates a payment through the simulated network, returning a tracking
    /// channel that can be used to obtain the result of the payment. At present, MPP payments are not supported.
    /// In future, we'll allow multiple paths for a single payment, so we allow the trait to accept a route with
    /// multiple paths to avoid future refactoring.
    fn dispatch_payment(
        &mut self,
        source: PublicKey,
        route: Route,
        preimage: PaymentPreimage,
        sender: Sender<Result<PaymentResult, LightningError>>,
    ) {
        // Expect at least one path (right now), with the intention to support multiple in future.
        let path = match route.paths.first() {
            Some(p) => p,
            None => {
                log::warn!("Find route did not return expected number of paths.");

                if let Err(e) = sender.send(Ok(PaymentResult {
                    htlc_count: 0,
                    payment_outcome: PaymentOutcome::RouteNotFound,
                })) {
                    log::error!("Could not send payment result: {:?}.", e);
                }

                return;
            }
        };

        self.tasks.spawn(propagate_payment(
            self.channels.clone(),
            source,
            path.clone(),
            preimage,
            sender,
            self.shutdown_trigger.clone(),
        ));
    }

    /// lookup_node fetches a node's information and channel capacities.
    async fn lookup_node(&self, node: &PublicKey) -> Result<(NodeInfo, Vec<u64>), LightningError> {
        match self.nodes.get(node) {
            Some(channels) => Ok((node_info(*node), channels.clone())),
            None => Err(LightningError::GetNodeInfoError(
                "Node not found".to_string(),
            )),
        }
    }
}

/// Adds htlcs to the simulation state along the path provided. Returning the index in the path from which to fail
/// back htlcs (if any) and a forwading error if the payment is not successfully added to the entire path.
///
/// For each hop in the route, we check both the addition of the HTLC and whether we can forward it. Take an example
/// route A --> B --> C, we will add this in two hops: A --> B then B -->C. For each hop, using A --> B as an example:
/// * Check whether A can add the outgoing HTLC (checks liquidity and in-flight restrictions).
///   * If no, fail the HTLC.
///   * If yes, add outgoing HTLC to A's channel.
/// * Check whether B will accept the forward.
///   * If no, fail the HTLC.
///   * If yes, continue to the next hop.
///
/// If successfully added to A --> B, this check will be repeated for B --> C.
///
/// Note that we don't have any special handling for the receiving node, once we've successfully added a outgoing HTLC
/// for the outgoing channel that is connected to the receiving node we'll return. To add invoice-related handling,
/// we'd need to include some logic that then decides whether to settle/fail the HTLC at the last hop here.
async fn add_htlcs(
    nodes: Arc<Mutex<HashMap<u64, SimulatedChannel>>>,
    source: PublicKey,
    route: Path,
    payment_hash: PaymentHash,
) -> Result<(), (Option<usize>, ForwardingError)> {
    let mut outgoing_node = source;
    let mut outgoing_amount = route.fee_msat() + route.final_value_msat();
    let mut outgoing_cltv = route
        .hops
        .iter()
        .fold(0, |sum, value| sum + value.cltv_expiry_delta);

    let mut fail_idx = None;

    for (i, hop) in route.hops.iter().enumerate() {
        let mut node_lock = nodes.lock().await;
        match node_lock.get_mut(&hop.short_channel_id) {
            Some(channel) => {
                if let Err(e) = channel.add_htlc(
                    outgoing_node,
                    Htlc {
                        amount_msat: outgoing_amount,
                        cltv_expiry: outgoing_cltv,
                        hash: payment_hash,
                    },
                ) {
                    // If we couldn't add to this HTLC, we only need to fail back from the preceeding hop, so we don't
                    // have to progress our fail_idx.
                    return Err((fail_idx, e));
                }

                // If the HTLC was successfully added, then we'll need to remove the HTLC from this channel if we fail,
                // so we progress our failure index to include this node.
                fail_idx = Some(i);

                // Once we've added the HTLC on this hop's channel, we want to check whether it has sufficient fee
                // and CLTV delta per the _next_ channel's policy (because fees and CLTV delta in LN are charged on
                // the outgoing link). We check the policy belonging to the node that we just forwarded to, which
                // represents the fee in that direction.
                //
                // Note that we don't check the final hop's requirements for CLTV delta at present.
                if i != route.hops.len() - 1 {
                    if let Some(channel) = node_lock.get(&route.hops[i + 1].short_channel_id) {
                        if let Err(e) = channel.check_htlc_forward(
                            hop.pubkey,
                            hop.cltv_expiry_delta,
                            outgoing_amount - hop.fee_msat,
                            hop.fee_msat,
                        ) {
                            // If we haven't met forwarding conditions for the next channel's policy, then we fail at
                            // the current index, because we've already added the HTLC as outgoing.
                            return Err((fail_idx, e));
                        }
                    }
                }
            }
            None => {
                return Err((
                    fail_idx,
                    ForwardingError::ChannelNotFound(hop.short_channel_id),
                ))
            }
        }

        // Once we've taken the "hop" to the destination pubkey, it becomes the source of the next outgoing htlc.
        outgoing_node = hop.pubkey;
        outgoing_amount -= hop.fee_msat;
        outgoing_cltv -= hop.cltv_expiry_delta;

        // TODO: introduce artificial latency between hops?
    }

    Ok(())
}

/// Removes htlcs from the simulation state from the index in the path provided (backwards).
///
/// Taking the example of a payment over A --> B --> C --> D where the payment was rejected by C because it did not
/// have enough liquidity to forward it, we will expect a failure index of 1 because the HTLC was successfully added
/// to A and B's outgoing channels, but not C.
///
/// This function will remove the HTLC one hop at a time, working backwards from the failure index, so in this
/// case B --> C and then B --> A. We lookup the HTLC on the incoming node because it will have tracked it in its
/// outgoing in-flight HTLCs.
async fn remove_htlcs(
    nodes: Arc<Mutex<HashMap<u64, SimulatedChannel>>>,
    resolution_idx: usize,
    source: PublicKey,
    route: Path,
    payment_hash: PaymentHash,
    success: bool,
) -> Result<(), ForwardingError> {
    for i in (0..resolution_idx).rev() {
        let hop = &route.hops[i];

        // When we add HTLCs, we do so on the state of the node that sent the htlc along the channel so we need to
        // look up our incoming node so that we can remove it when we go backwards. For the first htlc, this is just
        // the sending node, otherwise it's the hop before.
        let incoming_node = if i == 0 {
            source
        } else {
            route.hops[i - 1].pubkey
        };

        match nodes.lock().await.get_mut(&hop.short_channel_id) {
            Some(channel) => channel.remove_htlc(incoming_node, payment_hash, success)?,
            None => return Err(ForwardingError::ChannelNotFound(hop.short_channel_id)),
        }
    }

    Ok(())
}

/// Finds a payment path from the source to destination nodes provided, and propagates the appropriate htlcs through
/// the simulated network, notifying the sender channel provided of the payment outcome. If a critical error occurs,
/// ie a breakdown of our state machine, it will still notify the payment outcome and will use the shutdown trigger
/// to signal that we should exit.
async fn propagate_payment(
    nodes: Arc<Mutex<HashMap<u64, SimulatedChannel>>>,
    source: PublicKey,
    route: Path,
    preimage: PaymentPreimage,
    sender: Sender<Result<PaymentResult, LightningError>>,
    shutdown: Trigger,
) {
    let preimage_bytes = Sha256::hash(&preimage.0[..]).to_byte_array();
    let payment_hash = PaymentHash(preimage_bytes);

    let notify_result = match add_htlcs(nodes.clone(), source, route.clone(), payment_hash).await {
        // If we successfully added the htlc, go ahead and remove all the htlcs in the route with successful resolution.
        Ok(_) => {
            if let Err(e) = remove_htlcs(
                nodes,
                route.hops.len() - 1,
                source,
                route,
                payment_hash,
                true,
            )
            .await
            {
                if e.is_critical() {
                    shutdown.trigger();
                }

                log::error!("Could not remove htlcs from channel: {e}.");
            }

            PaymentResult {
                htlc_count: 1,
                payment_outcome: PaymentOutcome::Success,
            }
        }
        // If we partially added HTLCs along the route, we need to fail them back to the source to clean up our
        // partial state. It's possible that we failed with the very first add, and then we don't need to clean
        // anything up.
        Err((fail_idx, err)) => {
            if err.is_critical() {
                shutdown.trigger();
            }

            if let Some(resolution_idx) = fail_idx {
                if let Err(e) =
                    remove_htlcs(nodes, resolution_idx, source, route, payment_hash, false).await
                {
                    if e.is_critical() {
                        shutdown.trigger();
                    }
                }
            }

            // We have more information about failures because we're in control of the whole route, so we log the
            // actual failure reason and then fail back with unknown failure type.
            log::debug!(
                "Forwarding failure for simulated payment {}: {err}",
                hex::encode(payment_hash.0)
            );

            PaymentResult {
                htlc_count: 0,
                payment_outcome: PaymentOutcome::Unknown,
            }
        }
    };

    if let Err(e) = sender.send(Ok(notify_result)) {
        log::error!("Could not notify payment result: {:?}.", e);
    }
}

/// WrappedLog implements LDK's logging trait so that we can provide pathfinding with a logger that uses our existing
/// logger.
pub struct WrappedLog {}

impl Logger for WrappedLog {
    fn log(&self, record: Record) {
        match record.level {
            Level::Gossip => log::trace!("{}", record.args),
            Level::Trace => log::trace!("{}", record.args),
            Level::Debug => log::debug!("{}", record.args),
            Level::Info => log::debug!("{}", record.args),
            Level::Warn => log::warn!("{}", record.args),
            Level::Error => log::error!("{}", record.args),
        }
    }
}

/// UtxoValidator is a mock utxo validator that just returns a fake output with the desired capacity for a channel.
struct UtxoValidator {
    amount_sat: u64,
    script: ScriptBuf,
}

impl UtxoLookup for UtxoValidator {
    fn get_utxo(&self, _genesis_hash: &ChainHash, _short_channel_id: u64) -> UtxoResult {
        UtxoResult::Sync(Ok(TxOut {
            value: self.amount_sat,
            script_pubkey: self.script.clone(),
        }))
    }
}
