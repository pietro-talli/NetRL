// Minimal ns-3 side bridge skeleton for NetRL.
//
// This file is a starting point showing where to wire ns-3 simulation
// events into a TCP JSON bridge. It does not include the full ns-3
// application boilerplate.

#include <string>

namespace netrl {

struct Observation {
  std::string payload_json;
};

struct Action {
  std::string payload_json;
};

class NetRlBridge {
 public:
  explicit NetRlBridge(const std::string &host, int port)
      : host_(host), port_(port) {}

  Observation BuildObservation() {
    // TODO: Extract state from ns-3 simulation and serialize to JSON.
    return Observation{"{\"state\": \"todo\"}"};
  }

  void ApplyAction(const Action &action) {
    // TODO: Parse JSON action and apply to ns-3 simulation.
    (void)action;
  }

 private:
  std::string host_;
  int port_;
};

}  // namespace netrl
