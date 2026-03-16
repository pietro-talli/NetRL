/**
 * ns3_wifi_sim.cc
 *
 * Standalone ns-3 WiFi simulation program used as a subprocess by
 * netrl.NS3WifiChannel.
 *
 * Topology
 * --------
 *   [STA (agent node)] ──── 802.11a ad-hoc ────> [AP (central node)]
 *
 * The STA sends one UDP packet per env step. ns-3 simulates:
 *   - 802.11a PHY (OFDM, 5 GHz)
 *   - CSMA/CA MAC with configurable retry limit
 *   - Log-distance path loss
 *   - Constant-speed propagation delay
 *
 * Protocol (stdin / stdout, line-oriented)
 * -----------------------------------------
 * Python → program:
 *   TRANSMIT <step_id> <pkt_size>  schedule a probe packet send for step step_id
 *   FLUSH    <step_id>     advance sim to end of step_id, report arrivals
 *   RESET                  destroy & rebuild simulation (sim time → 0)
 *   QUIT                   graceful exit
 *
 * Program → Python:
 *   READY                  emitted once at startup (sim is ready)
 *   OK                     response to TRANSMIT / RESET
 *   RECV <id1> <id2> ...   response to FLUSH — space-separated step_ids
 *                          that arrived in this flush window (may be empty)
 *   ERROR <msg>            unexpected condition
 *
 * Timing model
 * ------------
 * Step t occupies ns-3 simulation time [t * step_ms, (t+1) * step_ms).
 * Each packet is sent at t * step_ms + 0.01 * step_ms (1% into the step).
 * FLUSH t advances the simulator to (t+1) * step_ms and collects all
 * packets whose receive callback fired during that window.
 *
 * Persistence
 * -----------
 * The ns-3 simulation runs continuously across env steps; Simulator::Run()
 * is called once per FLUSH with an increasing Stop time.  The full simulator
 * state (MAC buffers, backoff counters, etc.) persists between steps.
 * Simulator::Destroy() is called only on RESET (i.e. env.reset()).
 *
 * Build
 * -----
 *   cd /path/to/NetRL && bash src/build_ns3_sim.sh
 *
 * Usage
 * -----
 *   ./src/ns3_wifi_sim [options]
 *   Options:
 *     --step-ms=<float>   Step duration in milliseconds   (default: 1.0)
 *     --distance=<float>  STA–AP distance in metres       (default: 10.0)
 *     --tx-power=<float>  TX power in dBm                 (default: 20.0)
 *     --loss-exp=<float>  Log-distance path-loss exponent (default: 3.0)
 *     --retries=<int>     Max MAC retry count             (default: 7)
 *     --pkt-size=<int>    Probe packet payload bytes      (default: 64)
 */

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <deque>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

// ---------------------------------------------------------------------------
// Configuration (set from command line, constant for the life of the process)
// ---------------------------------------------------------------------------
static double g_stepMs    = 1.0;
static double g_distanceM = 10.0;
static double g_txPowerDbm = 20.0;
static double g_lossExp   = 3.0;
static int    g_maxRetries = 7;
static int    g_pktSize   = 64;   // bytes (payload only)

// ---------------------------------------------------------------------------
// Simulation state (rebuilt on RESET)
// ---------------------------------------------------------------------------
static NodeContainer       g_nodes;
static Ptr<Socket>         g_sendSocket;
static Ptr<Socket>         g_recvSocket;
static std::vector<uint32_t> g_arrivedIds;  // step IDs received in current flush window

// Reference point: simulation time (ms) that corresponds to env step 0.
// Set by BuildTopology() after the IBSS warm-up pre-run.
// All step-relative absolute times are: g_simStartMs + step * g_stepMs.
static double g_simStartMs = 0.0;

// ---------------------------------------------------------------------------
// Receive callback — fires inside Simulator::Run()
// ---------------------------------------------------------------------------
static void ReceivePacket(Ptr<Socket> socket)
{
    Ptr<Packet> pkt;
    Address     from;
    while ((pkt = socket->RecvFrom(from)) != nullptr) {
        uint32_t sz = pkt->GetSize();
        if (sz >= 4) {
            uint8_t buf[4];
            pkt->CopyData(buf, 4);
            uint32_t step_id = (static_cast<uint32_t>(buf[0]) << 24)
                             | (static_cast<uint32_t>(buf[1]) << 16)
                             | (static_cast<uint32_t>(buf[2]) << 8)
                             |  static_cast<uint32_t>(buf[3]);
            g_arrivedIds.push_back(step_id);
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduled send — fired by the simulator at the right sim time
// ---------------------------------------------------------------------------
static void DoSend(uint32_t step_id, int pkt_size)
{
    // Build 4-byte big-endian payload carrying the step_id
    uint8_t buf[4];
    buf[0] = static_cast<uint8_t>((step_id >> 24) & 0xFF);
    buf[1] = static_cast<uint8_t>((step_id >> 16) & 0xFF);
    buf[2] = static_cast<uint8_t>((step_id >>  8) & 0xFF);
    buf[3] = static_cast<uint8_t>( step_id        & 0xFF);

    // Pad to the per-packet requested size (minimum 4 bytes for the step_id)
    std::vector<uint8_t> payload(std::max(pkt_size, 4), 0);
    std::copy(buf, buf + 4, payload.begin());

    Ptr<Packet> pkt = Create<Packet>(payload.data(),
                                     static_cast<uint32_t>(payload.size()));
    g_sendSocket->Send(pkt);
}

// ---------------------------------------------------------------------------
// Build (or rebuild) the WiFi topology
// ---------------------------------------------------------------------------
static void BuildTopology()
{
    // Two nodes: 0 = STA (agent), 1 = AP (central node)
    g_nodes.Create(2);

    // --- WiFi PHY & channel ---
    YansWifiChannelHelper channelHelper;
    channelHelper.SetPropagationDelay(
        "ns3::ConstantSpeedPropagationDelayModel");
    channelHelper.AddPropagationLoss(
        "ns3::LogDistancePropagationLossModel",
        "Exponent",         DoubleValue(g_lossExp),
        "ReferenceDistance", DoubleValue(1.0),
        // Reference loss at 1m for 5 GHz (free-space, Friis)
        "ReferenceLoss",    DoubleValue(46.6777));

    YansWifiPhyHelper phyHelper;
    phyHelper.SetChannel(channelHelper.Create());
    phyHelper.Set("TxPowerStart", DoubleValue(g_txPowerDbm));
    phyHelper.Set("TxPowerEnd",   DoubleValue(g_txPowerDbm));

    // --- WiFi MAC (ad-hoc: no association delay) ---
    WifiMacHelper macHelper;
    macHelper.SetType("ns3::AdhocWifiMac");

    WifiHelper wifiHelper;
    wifiHelper.SetStandard(WIFI_STANDARD_80211a);
    wifiHelper.SetRemoteStationManager(
        "ns3::ConstantRateWifiManager",
        "DataMode",    StringValue("OfdmRate54Mbps"),
        "ControlMode", StringValue("OfdmRate6Mbps"));

    NetDeviceContainer devices =
        wifiHelper.Install(phyHelper, macHelper, g_nodes);

    // Set MAC-layer frame retry limit (ns-3 >= 3.43 uses FrameRetryLimit)
    Config::Set(
        "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/FrameRetryLimit",
        UintegerValue(static_cast<uint32_t>(g_maxRetries)));

    // --- Mobility (static positions) ---
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> posAlloc =
        CreateObject<ListPositionAllocator>();
    posAlloc->Add(Vector(0.0, 0.0, 0.0));              // STA
    posAlloc->Add(Vector(g_distanceM, 0.0, 0.0));      // AP
    mobility.SetPositionAllocator(posAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(g_nodes);

    // --- Internet (IPv4 + UDP) ---
    InternetStackHelper internet;
    internet.Install(g_nodes);

    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer iface = ipv4.Assign(devices);

    // --- Sockets ---
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");

    // Receive socket on AP (node 1)
    g_recvSocket = Socket::CreateSocket(g_nodes.Get(1), tid);
    g_recvSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), 9));
    g_recvSocket->SetRecvCallback(MakeCallback(&ReceivePacket));

    // Send socket on STA (node 0), connected to AP
    g_sendSocket = Socket::CreateSocket(g_nodes.Get(0), tid);
    InetSocketAddress apAddr(iface.GetAddress(1), 9);
    g_sendSocket->Connect(apAddr);

    // --- IBSS warm-up pre-run ---
    // In ns3 ad-hoc (IBSS) mode, nodes need to exchange beacon frames before
    // they will accept each other's data frames.  The default beacon interval
    // is ~102 ms.  We pre-run for 3 beacon intervals so that both nodes have
    // discovered each other and merged into the same BSS before step 0 begins.
    // g_simStartMs marks the ns3 clock time that corresponds to env step 0.
    const double warmupMs = g_stepMs * 2.0 < 310.0 ? 310.0 : g_stepMs * 2.0;
    Simulator::Stop(MilliSeconds(warmupMs));
    Simulator::Run();
    g_simStartMs = warmupMs;
}

// ---------------------------------------------------------------------------
// Full reset: destroy all ns3 state and rebuild from scratch
// ---------------------------------------------------------------------------
static void ResetSimulation()
{
    Simulator::Destroy();
    g_nodes       = NodeContainer();
    g_sendSocket  = nullptr;
    g_recvSocket  = nullptr;
    g_arrivedIds.clear();
    BuildTopology();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // Ignore SIGPIPE so we exit cleanly if Python closes the pipe
    signal(SIGPIPE, SIG_IGN);

    // Silence all ns3 log output (it uses std::clog/stderr anyway,
    // but this prevents any accidentally-enabled components from
    // polluting our line protocol on stdout).
    // Users can re-enable via NS_LOG env var for debugging.
    LogComponentDisableAll(LOG_LEVEL_ALL);

    // --- Parse command-line arguments ---
    CommandLine cmd(__FILE__);
    cmd.AddValue("step-ms",   "Step duration in milliseconds",       g_stepMs);
    cmd.AddValue("distance",  "STA–AP distance in metres",           g_distanceM);
    cmd.AddValue("tx-power",  "TX power in dBm",                     g_txPowerDbm);
    cmd.AddValue("loss-exp",  "Log-distance path-loss exponent",     g_lossExp);
    cmd.AddValue("retries",   "Max MAC retry count",                  g_maxRetries);
    cmd.AddValue("pkt-size",  "Probe packet payload bytes",           g_pktSize);
    cmd.Parse(argc, argv);

    // --- Build initial topology ---
    BuildTopology();

    // Signal Python that the simulation is ready
    std::cout << "READY" << std::endl;

    // --- Command loop ---
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string        command;
        iss >> command;

        // ---- TRANSMIT <step_id> <pkt_size> --------------------------------
        if (command == "TRANSMIT") {
            uint32_t step_id = 0;
            int      pkt_size = g_pktSize;
            iss >> step_id >> pkt_size;

            // Compute absolute ns3 send time: 1% into the env step window.
            // Step t occupies [g_simStartMs + t*step_ms,
            //                  g_simStartMs + (t+1)*step_ms).
            double sendAbsMs = g_simStartMs + step_id * g_stepMs + g_stepMs * 0.01;
            double nowMs     = Simulator::Now().GetMilliSeconds();
            double delayMs   = sendAbsMs - nowMs;

            if (delayMs > 0.0) {
                Simulator::Schedule(MilliSeconds(delayMs), &DoSend, step_id, pkt_size);
            } else {
                // Already past the absolute time (e.g. retransmit of old step)
                // Schedule with the smallest positive delay
                Simulator::Schedule(NanoSeconds(1), &DoSend, step_id, pkt_size);
            }

            std::cout << "OK" << std::endl;

        // ---- FLUSH <step_id> -------------------------------------------
        } else if (command == "FLUSH") {
            uint32_t step_id = 0;
            iss >> step_id;

            double endAbsMs  = g_simStartMs + (step_id + 1.0) * g_stepMs;
            double nowMs     = Simulator::Now().GetMilliSeconds();
            double delayMs   = endAbsMs - nowMs;

            g_arrivedIds.clear();

            if (delayMs > 0.0) {
                Simulator::Stop(MilliSeconds(delayMs));
                Simulator::Run();
            }
            // If already past endAbsMs, no Run() needed — nothing new will arrive

            // Build response line
            std::ostringstream resp;
            resp << "RECV";
            for (uint32_t sid : g_arrivedIds) {
                resp << ' ' << sid;
            }
            std::cout << resp.str() << std::endl;

        // ---- RESET -------------------------------------------------------
        } else if (command == "RESET") {
            ResetSimulation();
            std::cout << "OK" << std::endl;

        // ---- QUIT -------------------------------------------------------
        } else if (command == "QUIT") {
            break;

        // ---- Unknown ----------------------------------------------------
        } else {
            std::cout << "ERROR unknown command: " << command << std::endl;
        }
    }

    Simulator::Destroy();
    return 0;
}
