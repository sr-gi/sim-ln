use bitcoin::secp256k1::PublicKey;
use lightning::ln::PaymentHash;
use std::collections::HashMap;

/// ForwardingError represents the various errors that we can run into when forwarding payments in a simulated network.
/// Since we're not using real lightning nodes, these errors are not obfuscated and can be propagated to the sending
/// node and used for analysis.
#[derive(Debug)]
pub enum ForwardingError {
    /// Zero amount htlcs are in valid in the protocol.
    ZeroAmountHtlc,
    /// The outgoing channel id was not found in the network graph.
    ChannelNotFound(u64),
    /// The node pubkey provided was not associated with the channel in the network graph.
    NodeNotFound(PublicKey),
    /// The channel has already forwarded a HTLC with the payment hash provided (to be removed when MPP support is
    /// added).
    PaymentHashExists(PaymentHash),
    /// A htlc with the payment hash provided could not be found to resolve.
    PaymentHashNotFound(PaymentHash),
    /// The forwarding node did not have sufficient outgoing balance to forward the htlc (htlc amount / balance).
    InsufficientBalance(u64, u64),
    /// The htlc forwarded is less than the channel's advertised minimum htlc amount (htlc amount / minimum).
    LessThanMinimum(u64, u64),
    /// The htlc forwarded is more than the chanenl's advertised maximum htlc amount (htlc amount / maximum).
    MoreThanMaximum(u64, u64),
    /// The channel has reached its maximum allowable number of htlcs in flight (total in flight / maximim).
    ExceedsInFlightCount(u64, u64),
    /// The forwarded htlc's amount would push the channel over its maximum allowable in flight total
    /// (total in flight / maximum).
    ExceedsInFlightTotal(u64, u64),
    /// The forwarded htlc's cltv expiry exceeds the maximum value used to express block heights in bitcoin.
    ExpiryInSeconds(u32),
    /// The forwarded htlc has insufficient cltv delta for the channel's minimum delta (cltv delta / minimum).
    InsufficientCltvDelta(u32, u32),
    /// The forwarded htlc has insufficient fee for the channel's policy (fee / base fee / prop fee / expected fee).
    InsufficientFee(u64, u64, u64, u64),
    /// Sanity check on channel balances failed (capacity / node_1 balance / node_2 balance).
    SanityCheckFailed(u64, u64, u64),
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
