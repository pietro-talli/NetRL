/**
 * ns3_wifi_multi_ue_sim.cc
 *
 * Standalone ns-3 WiFi simulation: N UEs (STAs) → single AP (central node).
 * Used as a subprocess by netrl.NS3WifiUEChannel / NS3WifiMultiUEBackend.
 *
 * Topology
 * --------
 *   [UE_0 (STA)] ──┐
 *   [UE_1 (STA)] ──┤── 802.11a infrastructure BSS ──> [AP (central node)]
 *   [UE_N (STA)] ──┘
 *
 * Each UE sends their observation independently; all share the same wireless
 * medium and contend via CSMA/CA.  The AP aggregates all uplink packets.
 *
 * Protocol (stdin / stdout, line-oriented)
 * -----------------------------------------
 * Python → program:
 *   TRANSMIT <ue_id> <step_id> <pkt_size>   schedule packet from UE ue_id
 *   FLUSH    <step_id>    advance sim to end of step, report all arrivals
 *   RESET                 destroy & rebuild simulation (sim time → 0)
 *   QUIT                  graceful exit
 *
 * Program → Python:
 *   READY                         emitted once at startup (after association)
 *   OK                            response to TRANSMIT / RESET
 *   RECV <ue_id>:<step_id> ...    response to FLUSH — space-separated pairs
 *                                 (may be empty: "RECV")
 *   ERROR <msg>                   unexpected condition
 *
 * Timing model
 * ------------
 * Step t occupies ns-3 simulation time [t * step_ms, (t+1) * step_ms).
 * UE ue_id sends its packet at t * step_ms + (0.01 + ue_id * 0.002) * step_ms
 * to avoid simultaneous MAC contention at the start of each step.
 * FLUSH t advances the simulator to (t+1) * step_ms and collects all
 * packets whose receive callback fired during that window.
 *
 * Build
 * -----
 *   bash src/build_ns3_multi_ue_sim.sh
 *
 * Usage
 * -----
 *   ./src/ns3_wifi_multi_ue_sim \
 *     --n-ues=3              \
 *     --distances=10.0,20.0,30.0 \
 *     --step-ms=1.0          \
 *     --tx-power=20.0        \
 *     --loss-exp=3.0         \
 *     --retries=7            \
 *     --pkt-size=64
 */

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace ns3;

// ---------------------------------------------------------------------------
// Configuration (set from command-line, constant for the process lifetime)
// ---------------------------------------------------------------------------
static int         g_nUes         = 2;
static std::string g_distancesStr = "10.0";  // comma-separated per-UE distances
static double      g_stepMs       = 1.0;
static double      g_txPowerDbm   = 20.0;
static double      g_lossExp      = 3.0;
static int         g_maxRetries   = 7;
static int         g_pktSize      = 64;

// Parsed from g_distancesStr after cmd.Parse()
static std::vector<double> g_distances;

// ---------------------------------------------------------------------------
// Simulation state (rebuilt on every RESET)
// ---------------------------------------------------------------------------
static NodeContainer             g_nodes;          // node 0 = AP; 1..N = STAs
static std::vector<Ptr<Socket>>  g_sendSockets;    // one per UE (indexed by ue_id)
static Ptr<Socket>               g_recvSocket;

// Arrivals recorded during the current flush window: (ue_id, step_id)
static std::vector<std::pair<uint32_t, uint32_t>> g_arrivedPairs;

// ns-3 simulation time (ms) that corresponds to env step 0, set after warm-up
static double g_simStartMs = 0.0;

// ---------------------------------------------------------------------------
// Helper: parse g_distancesStr into g_distances
// ---------------------------------------------------------------------------
static void ParseDistances()
{
    g_distances.clear();
    std::istringstream ss(g_distancesStr);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        try {
            g_distances.push_back(std::stod(token));
        } catch (...) {
            g_distances.push_back(10.0);
        }
    }
    if (g_distances.empty()) {
        g_distances.push_back(10.0);
    }
    // Repeat last value if fewer distances than UEs
    while (static_cast<int>(g_distances.size()) < g_nUes) {
        g_distances.push_back(g_distances.back());
    }
}

// ---------------------------------------------------------------------------
// Receive callback — fires inside Simulator::Run() when AP receives a packet
// ---------------------------------------------------------------------------
static void ReceivePacket(Ptr<Socket> socket)
{
    Ptr<Packet> pkt;
    Address     from;
    while ((pkt = socket->RecvFrom(from)) != nullptr) {
        if (pkt->GetSize() >= 8) {
            uint8_t buf[8];
            pkt->CopyData(buf, 8);
            uint32_t ue_id   = (static_cast<uint32_t>(buf[0]) << 24)
                             | (static_cast<uint32_t>(buf[1]) << 16)
                             | (static_cast<uint32_t>(buf[2]) <<  8)
                             |  static_cast<uint32_t>(buf[3]);
            uint32_t step_id = (static_cast<uint32_t>(buf[4]) << 24)
                             | (static_cast<uint32_t>(buf[5]) << 16)
                             | (static_cast<uint32_t>(buf[6]) <<  8)
                             |  static_cast<uint32_t>(buf[7]);
            g_arrivedPairs.push_back({ue_id, step_id});
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduled send — fired by the simulator scheduler at the correct sim time
// ---------------------------------------------------------------------------
static void DoSend(uint32_t ue_id, uint32_t step_id, int pkt_size)
{
    // 8-byte big-endian header: [ue_id (4 B)][step_id (4 B)]
    uint8_t hdr[8];
    hdr[0] = static_cast<uint8_t>((ue_id   >> 24) & 0xFF);
    hdr[1] = static_cast<uint8_t>((ue_id   >> 16) & 0xFF);
    hdr[2] = static_cast<uint8_t>((ue_id   >>  8) & 0xFF);
    hdr[3] = static_cast<uint8_t>( ue_id          & 0xFF);
    hdr[4] = static_cast<uint8_t>((step_id >> 24) & 0xFF);
    hdr[5] = static_cast<uint8_t>((step_id >> 16) & 0xFF);
    hdr[6] = static_cast<uint8_t>((step_id >>  8) & 0xFF);
    hdr[7] = static_cast<uint8_t>( step_id        & 0xFF);

    std::vector<uint8_t> payload(std::max(pkt_size, 8), 0);
    std::copy(hdr, hdr + 8, payload.begin());

    Ptr<Packet> pkt = Create<Packet>(payload.data(),
                                     static_cast<uint32_t>(payload.size()));
    if (ue_id < g_sendSockets.size()) {
        g_sendSockets[ue_id]->Send(pkt);
    }
}

// ---------------------------------------------------------------------------
// Build (or rebuild) the infrastructure WiFi topology
// ---------------------------------------------------------------------------
static void BuildTopology()
{
    // node 0 = AP, nodes 1..n_ues = STAs
    g_nodes.Create(static_cast<uint32_t>(g_nUes + 1));

    // --- Shared wireless channel ---
    YansWifiChannelHelper channelHelper;
    channelHelper.SetPropagationDelay(
        "ns3::ConstantSpeedPropagationDelayModel");
    channelHelper.AddPropagationLoss(
        "ns3::LogDistancePropagationLossModel",
        "Exponent",          DoubleValue(g_lossExp),
        "ReferenceDistance", DoubleValue(1.0),
        "ReferenceLoss",     DoubleValue(46.6777));  // 5 GHz free-space at 1 m

    YansWifiPhyHelper phyHelper;
    phyHelper.SetChannel(channelHelper.Create());
    phyHelper.Set("TxPowerStart", DoubleValue(g_txPowerDbm));
    phyHelper.Set("TxPowerEnd",   DoubleValue(g_txPowerDbm));

    WifiHelper wifiHelper;
    wifiHelper.SetStandard(WIFI_STANDARD_80211a);
    wifiHelper.SetRemoteStationManager(
        "ns3::ConstantRateWifiManager",
        "DataMode",    StringValue("OfdmRate54Mbps"),
        "ControlMode", StringValue("OfdmRate6Mbps"));

    // --- Infrastructure BSS: AP ---
    Ssid ssid = Ssid("netrl-bss");
    WifiMacHelper macAP;
    macAP.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer apDevice =
        wifiHelper.Install(phyHelper, macAP, g_nodes.Get(0));

    // --- Infrastructure BSS: STAs (one per UE) ---
    // ActiveProbing=true: STA immediately sends a ProbeRequest rather than
    // waiting for a beacon, cutting association time from ~100 ms to ~5 ms.
    WifiMacHelper macSTA;
    macSTA.SetType("ns3::StaWifiMac",
                   "Ssid",           SsidValue(ssid),
                   "ActiveProbing",  BooleanValue(true));
    NodeContainer staNodes;
    for (int i = 0; i < g_nUes; ++i) {
        staNodes.Add(g_nodes.Get(static_cast<uint32_t>(i + 1)));
    }
    NetDeviceContainer staDevices =
        wifiHelper.Install(phyHelper, macSTA, staNodes);

    // MAC frame retry limit (applied to all nodes; AP never retries uplink)
    Config::Set(
        "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/FrameRetryLimit",
        UintegerValue(static_cast<uint32_t>(g_maxRetries)));

    // --- Static positions: AP at origin, each UE along the X-axis ---
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> posAlloc =
        CreateObject<ListPositionAllocator>();
    posAlloc->Add(Vector(0.0, 0.0, 0.0));  // AP
    for (int i = 0; i < g_nUes; ++i) {
        double d = (i < static_cast<int>(g_distances.size()))
                   ? g_distances[static_cast<std::size_t>(i)]
                   : g_distances.back();
        posAlloc->Add(Vector(d, 0.0, 0.0));
    }
    mobility.SetPositionAllocator(posAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(g_nodes);

    // --- Internet stack (IPv4 + UDP) ---
    InternetStackHelper internet;
    internet.Install(g_nodes);

    // Assign IPs: AP gets .1, STAs get .2, .3, ...
    NetDeviceContainer allDevices;
    allDevices.Add(apDevice);
    allDevices.Add(staDevices);
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer iface = ipv4.Assign(allDevices);

    // --- Sockets ---
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");

    // Receive socket on AP (node 0): collects packets from all UEs
    g_recvSocket = Socket::CreateSocket(g_nodes.Get(0), tid);
    g_recvSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), 9));
    g_recvSocket->SetRecvCallback(MakeCallback(&ReceivePacket));

    // Send sockets on STAs: one per UE, connected to the AP's IP
    Ipv4Address apAddr = iface.GetAddress(0);
    g_sendSockets.resize(static_cast<std::size_t>(g_nUes));
    for (int i = 0; i < g_nUes; ++i) {
        g_sendSockets[static_cast<std::size_t>(i)] =
            Socket::CreateSocket(g_nodes.Get(static_cast<uint32_t>(i + 1)), tid);
        g_sendSockets[static_cast<std::size_t>(i)]->Connect(
            InetSocketAddress(apAddr, 9));
    }

    // --- Infrastructure warm-up ---
    // STAs must complete probe/auth/association before sending data.
    // With ActiveProbing=true and N concurrent STAs using CSMA-CA backoff,
    // 500 ms comfortably covers association for up to ~20 UEs.
    const double warmupMs = 500.0;
    Simulator::Stop(MilliSeconds(warmupMs));
    Simulator::Run();
    g_simStartMs = warmupMs;
}

// ---------------------------------------------------------------------------
// Full reset: tear down all ns-3 state and rebuild from scratch
// ---------------------------------------------------------------------------
static void ResetSimulation()
{
    Simulator::Destroy();
    g_nodes = NodeContainer();
    g_sendSockets.clear();
    g_recvSocket  = nullptr;
    g_arrivedPairs.clear();
    BuildTopology();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    signal(SIGPIPE, SIG_IGN);
    LogComponentDisableAll(LOG_LEVEL_ALL);

    CommandLine cmd(__FILE__);
    cmd.AddValue("n-ues",      "Number of UEs / STAs",                   g_nUes);
    cmd.AddValue("distances",  "Comma-separated UE-to-AP distances (m)", g_distancesStr);
    cmd.AddValue("step-ms",    "Step duration in milliseconds",           g_stepMs);
    cmd.AddValue("tx-power",   "TX power in dBm",                        g_txPowerDbm);
    cmd.AddValue("loss-exp",   "Log-distance path-loss exponent",        g_lossExp);
    cmd.AddValue("retries",    "Max MAC retry count",                     g_maxRetries);
    cmd.AddValue("pkt-size",   "Probe packet payload bytes",              g_pktSize);
    cmd.Parse(argc, argv);

    ParseDistances();
    BuildTopology();

    std::cout << "READY" << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string        command;
        iss >> command;

        // ---- TRANSMIT <ue_id> <step_id> <pkt_size> -------------------------
        if (command == "TRANSMIT") {
            uint32_t ue_id   = 0;
            uint32_t step_id = 0;
            int      pkt_size = g_pktSize;
            iss >> ue_id >> step_id >> pkt_size;

            if (static_cast<int>(ue_id) >= g_nUes) {
                std::cout << "ERROR ue_id " << ue_id
                          << " out of range [0," << g_nUes << ")"
                          << std::endl;
                continue;
            }

            // Schedule send at: (1% + ue_id * 0.2%) into the step window.
            // The small per-UE offset avoids exact-simultaneous MAC contention
            // at the very start of each step while keeping all transmissions
            // well within the first 5% of the step.
            double offsetFrac = 0.0; // 0.01 + ue_id * 0.002; // TODO: Think about whether to re-enable this offset
            double sendAbsMs  = g_simStartMs + step_id * g_stepMs
                              + g_stepMs * offsetFrac;
            double nowMs      = Simulator::Now().GetMilliSeconds();
            double delayMs    = sendAbsMs - nowMs;

            if (delayMs > 0.0) {
                Simulator::Schedule(MilliSeconds(delayMs),
                                    &DoSend, ue_id, step_id, pkt_size);
            } else {
                Simulator::Schedule(NanoSeconds(1),
                                    &DoSend, ue_id, step_id, pkt_size);
            }

            std::cout << "OK" << std::endl;

        // ---- FLUSH <step_id> -----------------------------------------------
        } else if (command == "FLUSH") {
            uint32_t step_id = 0;
            iss >> step_id;

            double endAbsMs = g_simStartMs + (step_id + 1.0) * g_stepMs;
            double nowMs    = Simulator::Now().GetMilliSeconds();
            double delayMs  = endAbsMs - nowMs;

            g_arrivedPairs.clear();

            if (delayMs > 0.0) {
                Simulator::Stop(MilliSeconds(delayMs));
                Simulator::Run();
            }

            // Build "RECV ue_id:step_id ue_id:step_id ..." response
            std::ostringstream resp;
            resp << "RECV";
            for (const auto& p : g_arrivedPairs) {
                resp << ' ' << p.first << ':' << p.second;
            }
            std::cout << resp.str() << std::endl;

        // ---- RESET ---------------------------------------------------------
        } else if (command == "RESET") {
            ResetSimulation();
            std::cout << "OK" << std::endl;

        // ---- QUIT ----------------------------------------------------------
        } else if (command == "QUIT") {
            break;

        // ---- Unknown -------------------------------------------------------
        } else {
            std::cout << "ERROR unknown command: " << command << std::endl;
        }
    }

    Simulator::Destroy();
    return 0;
}
