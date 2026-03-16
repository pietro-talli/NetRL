/**
 * netcomm.cpp
 *
 * Gilbert-Elliott two-state Markov channel implemented in C++ with
 * pybind11 bindings.
 *
 * Channel model
 * -------------
 * Two states: GOOD (0) and BAD (1).
 *
 * At each transmit() call:
 *   1. Advance Markov state:
 *        GOOD -> BAD  with probability p_gb
 *        BAD  -> GOOD with probability p_bg
 *   2. Sample loss ~ Bernoulli(loss_prob[state]).
 *   3. If not lost, enqueue PendingPacket{arrival_step = step + delay_steps}.
 *
 * At each flush(step) call:
 *   Return (and discard) all queued packets with arrival_step <= step.
 *   Because delay is fixed, the queue is always in FIFO / arrival order,
 *   so pop from the front until arrival_step > step.
 *
 * Build
 * -----
 *   pip install pybind11 && pip install -e .
 *   # or directly:
 *   python setup.py build_ext --inplace
 *
 * Requirements: C++17, pybind11 >= 2.11
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <deque>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

enum class ChannelState : uint8_t { GOOD = 0, BAD = 1 };

/**
 * A packet that passed the loss filter and is waiting to be delivered.
 * The observation is stored as a flat vector<double> + original shape so
 * it can be reconstructed into a NumPy array on flush().
 */
struct PendingPacket {
    int                  arrival_step;
    std::vector<double>  data;
    std::vector<ssize_t> shape;
};

// ---------------------------------------------------------------------------
// GEChannelImpl
// ---------------------------------------------------------------------------

class GEChannelImpl {
public:
    /**
     * Construct a Gilbert-Elliott channel.
     *
     * Parameters
     * ----------
     * p_gb        : Prob(Good -> Bad) per step.
     * p_bg        : Prob(Bad -> Good) per step.
     * loss_good   : Packet loss probability in Good state.
     * loss_bad    : Packet loss probability in Bad state.
     * delay_steps : Fixed propagation delay in environment steps (>= 0).
     * seed        : RNG seed. 0 means use std::random_device (non-deterministic).
     */
    GEChannelImpl(
        double   p_gb,
        double   p_bg,
        double   loss_good,
        double   loss_bad,
        int      delay_steps,
        uint64_t seed = 0
    )
        : p_gb_(p_gb), p_bg_(p_bg),
          loss_good_(loss_good), loss_bad_(loss_bad),
          delay_steps_(delay_steps),
          state_(ChannelState::GOOD),
          rng_(seed == 0 ? std::random_device{}() : seed)
    {
        validate_params();
    }

    // -----------------------------------------------------------------------
    // transmit()
    //
    // Called once per env.step().
    // 1. Advances the Markov chain by one step.
    // 2. Samples whether the packet is lost.
    // 3. If not lost, copies the NumPy observation and enqueues a PendingPacket.
    // -----------------------------------------------------------------------
    void transmit(py::array_t<double, py::array::c_style | py::array::forcecast> obs, int step)
    {
        // Step 1: advance Markov state
        advance_state();

        // Step 2: sample loss
        double loss_p = (state_ == ChannelState::GOOD) ? loss_good_ : loss_bad_;
        std::bernoulli_distribution loss_dist(loss_p);
        bool lost = loss_dist(rng_);

        if (!lost) {
            // Step 3: copy obs and schedule delivery
            auto buf = obs.request();

            PendingPacket pkt;
            pkt.arrival_step = step + delay_steps_;
            pkt.data.assign(
                static_cast<double*>(buf.ptr),
                static_cast<double*>(buf.ptr) + buf.size
            );
            pkt.shape.assign(buf.shape.begin(), buf.shape.end());

            pending_.push_back(std::move(pkt));
        }
    }

    // -----------------------------------------------------------------------
    // flush()
    //
    // Called once per env.step() after transmit().
    // Returns all packets with arrival_step <= step as a list of
    // (arrival_step, ndarray) pairs. The queue is in FIFO order under
    // fixed delay, so we simply pop from the front.
    // -----------------------------------------------------------------------
    std::vector<std::pair<int, py::array_t<double>>> flush(int step)
    {
        std::vector<std::pair<int, py::array_t<double>>> result;

        while (!pending_.empty() && pending_.front().arrival_step <= step) {
            const PendingPacket& pkt = pending_.front();

            // Reconstruct NumPy array
            py::array_t<double> arr(pkt.shape);
            auto mutable_buf = arr.request();
            std::copy(
                pkt.data.begin(), pkt.data.end(),
                static_cast<double*>(mutable_buf.ptr)
            );

            result.emplace_back(pkt.arrival_step, std::move(arr));
            pending_.pop_front();
        }

        return result;
    }

    // -----------------------------------------------------------------------
    // reset()
    //
    // Clears pending queue and resets Markov state to GOOD.
    // The RNG is NOT re-seeded; construct a new instance to reseed.
    // -----------------------------------------------------------------------
    void reset()
    {
        pending_.clear();
        state_ = ChannelState::GOOD;
    }

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    std::string get_state_str() const
    {
        return (state_ == ChannelState::GOOD) ? "GOOD" : "BAD";
    }

    int get_pending_count() const
    {
        return static_cast<int>(pending_.size());
    }

    py::dict get_channel_info() const
    {
        py::dict d;
        d["state"]         = get_state_str();
        d["pending_count"] = get_pending_count();
        d["p_gb"]          = p_gb_;
        d["p_bg"]          = p_bg_;
        d["loss_good"]     = loss_good_;
        d["loss_bad"]      = loss_bad_;
        d["delay_steps"]   = delay_steps_;
        return d;
    }

private:
    // Advance the Markov chain by one step.
    void advance_state()
    {
        if (state_ == ChannelState::GOOD) {
            std::bernoulli_distribution d(p_gb_);
            if (d(rng_)) state_ = ChannelState::BAD;
        } else {
            std::bernoulli_distribution d(p_bg_);
            if (d(rng_)) state_ = ChannelState::GOOD;
        }
    }

    void validate_params()
    {
        auto check_prob = [](double v, const char* name) {
            if (v < 0.0 || v > 1.0)
                throw std::invalid_argument(
                    std::string(name) + " must be in [0, 1], got " +
                    std::to_string(v));
        };
        check_prob(p_gb_,      "p_gb");
        check_prob(p_bg_,      "p_bg");
        check_prob(loss_good_, "loss_good");
        check_prob(loss_bad_,  "loss_bad");
        if (delay_steps_ < 0)
            throw std::invalid_argument(
                "delay_steps must be >= 0, got " +
                std::to_string(delay_steps_));
    }

    // Channel parameters
    double p_gb_, p_bg_;
    double loss_good_, loss_bad_;
    int    delay_steps_;

    // Dynamic state
    ChannelState              state_;
    std::mt19937_64           rng_;
    std::deque<PendingPacket> pending_;
};

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(netcomm, m)
{
    m.doc() = R"doc(
netcomm — Gilbert-Elliott networked channel (C++ pybind11 backend).

The module exposes a single class, GEChannelImpl, which simulates a
two-state Markov (Gilbert-Elliott) channel with configurable loss
probabilities and a fixed propagation delay.

Typical usage (via the Python wrapper netrl.GEChannel):
    import netcomm
    ch = netcomm.GEChannelImpl(p_gb=0.1, p_bg=0.3,
                                loss_good=0.01, loss_bad=0.2,
                                delay_steps=3, seed=42)
    ch.transmit(obs_array, step=0)
    packets = ch.flush(step=3)   # -> [(3, obs_array)] if not lost
    ch.reset()
)doc";

    py::class_<GEChannelImpl>(m, "GEChannelImpl")
        .def(py::init<double, double, double, double, int, uint64_t>(),
             py::arg("p_gb"),
             py::arg("p_bg"),
             py::arg("loss_good"),
             py::arg("loss_bad"),
             py::arg("delay_steps"),
             py::arg("seed") = static_cast<uint64_t>(0),
             R"doc(
Gilbert-Elliott two-state Markov channel.

Parameters
----------
p_gb        : float   Transition probability Good -> Bad per step.
p_bg        : float   Transition probability Bad -> Good per step.
loss_good   : float   Packet loss probability in Good state.
loss_bad    : float   Packet loss probability in Bad state.
delay_steps : int     Fixed propagation delay in environment steps (>= 0).
seed        : int     RNG seed. 0 = non-deterministic (std::random_device).
)doc")
        .def("transmit", &GEChannelImpl::transmit,
             py::arg("obs"), py::arg("step"),
             R"doc(
Simulate transmission of one observation.

Advances the Markov state, samples loss, and if the packet is not lost
enqueues it with arrival_step = step + delay_steps.

Parameters
----------
obs  : np.ndarray[float64]  Flattened or shaped observation from env.step().
step : int                  Current integer step counter (0-indexed).
)doc")
        .def("flush", &GEChannelImpl::flush,
             py::arg("step"),
             R"doc(
Retrieve all packets due at or before `step`.

Returns
-------
List of (arrival_step: int, obs: np.ndarray) tuples.
Empty list if no packet is due this step.
)doc")
        .def("reset", &GEChannelImpl::reset,
             "Clear all in-flight packets and reset Markov state to GOOD. "
             "The RNG is NOT re-seeded.")
        .def("get_channel_info", &GEChannelImpl::get_channel_info,
             "Return a dict with diagnostic state: state, pending_count, "
             "p_gb, p_bg, loss_good, loss_bad, delay_steps.")
        .def_property_readonly("state",
             &GEChannelImpl::get_state_str,
             "Current Markov state string: 'GOOD' or 'BAD'.")
        .def_property_readonly("pending_count",
             &GEChannelImpl::get_pending_count,
             "Number of in-flight packets not yet delivered.");
}
