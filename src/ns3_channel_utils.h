/**
 * ns3_channel_utils.h
 *
 * Shared header-only utilities for NS3 channel subprocess simulators.
 * Eliminates the boilerplate that is duplicated across ns3_*_sim.cc files.
 *
 * What this covers
 * ----------------
 *   - NS3 initialisation (SIGPIPE + logging)
 *   - 4-byte big-endian step-id encode / decode
 *   - Probe packet creation
 *   - UDP socket helpers (recv + send)
 *   - IPv4 internet stack + address assignment
 *   - Static (ConstantPosition) mobility install
 *   - YANS 802.11a PHY + WifiHelper configuration
 *   - Warmup pre-run (beacon sync)
 *   - Full TRANSMIT / FLUSH / RESET / QUIT command loop
 *
 * What this does NOT cover
 * ------------------------
 *   - BuildTopology()   — topology is channel-specific; implement it yourself
 *   - ResetSimulation() — destroy + rebuild; call BuildTopology() inside it
 *   - Command-line argument parsing — each sim has its own parameters
 *   - 5G/EPC/mmWave/LENA helpers — those are module-specific
 *   - Multi-UE variant (8-byte header, per-UE sockets) — protocol differs
 *
 * Usage
 * -----
 *   // 1. Include this header AFTER all ns-3 module headers.
 *   #include "ns3_channel_utils.h"
 *
 *   // 2. Keep your own globals for channel parameters and NS3 objects.
 *   //    This header does not define any globals.
 *   static double g_stepMs = 1.0;
 *   static double g_simStartMs = 0.0;
 *   static std::vector<uint32_t> g_arrivedIds;
 *   static Ptr<Socket> g_sendSocket, g_recvSocket;
 *
 *   // 3. Implement the receive callback using DecodeStepId():
 *   static void ReceivePacket(Ptr<Socket> sock) {
 *       Ptr<Packet> p; Address from;
 *       while ((p = sock->RecvFrom(from))) {
 *           if (p->GetSize() < 4) continue;
 *           uint8_t buf[4]; p->CopyData(buf, 4);
 *           g_arrivedIds.push_back(netrl::DecodeStepId(buf));
 *       }
 *   }
 *
 *   // 4. Implement DoSend() using CreateProbePacket():
 *   static void DoSend(uint32_t step_id, int pkt_size) {
 *       g_sendSocket->Send(netrl::CreateProbePacket(step_id, pkt_size));
 *   }
 *
 *   // 5. In main(): initialise, parse args, build, then hand off to RunCommandLoop():
 *   int main(int argc, char* argv[]) {
 *       netrl::InitNS3();
 *       CommandLine cmd(__FILE__); // ... add values, parse ...
 *       BuildTopology();           // sets g_simStartMs
 *       netrl::RunCommandLoop(g_stepMs, g_simStartMs, g_arrivedIds,
 *                             &DoSend, &ResetSimulation, g_pktSize);
 *       return 0;
 *   }
 *
 * Note: RunCommandLoop() calls Simulator::Destroy() before it returns.
 */

#pragma once

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
#include <vector>

namespace netrl {

using namespace ns3;

// ============================================================================
// NS3 initialisation
// ============================================================================

/**
 * Call at the top of main() before anything else.
 *
 * - Ignores SIGPIPE so the process exits cleanly when Python closes the pipe.
 * - Silences all NS3 log components (stdout must stay clean for the protocol).
 *   Users can still enable specific components via the NS_LOG env-var.
 */
inline void InitNS3()
{
    signal(SIGPIPE, SIG_IGN);
    LogComponentDisableAll(LOG_LEVEL_ALL);
}

// ============================================================================
// Packet encode / decode
// ============================================================================

/**
 * Decode the first 4 bytes of a received packet as a big-endian step_id.
 *
 * Typical use inside a ReceivePacket callback:
 *   uint8_t buf[4]; pkt->CopyData(buf, 4);
 *   uint32_t step_id = netrl::DecodeStepId(buf);
 */
inline uint32_t DecodeStepId(const uint8_t* buf)
{
    return (static_cast<uint32_t>(buf[0]) << 24)
         | (static_cast<uint32_t>(buf[1]) << 16)
         | (static_cast<uint32_t>(buf[2]) <<  8)
         |  static_cast<uint32_t>(buf[3]);
}

/**
 * Encode step_id as a 4-byte big-endian vector.
 *
 * Typical use inside a DoSend function:
 *   auto hdr = netrl::EncodeStepId(step_id);
 */
inline std::vector<uint8_t> EncodeStepId(uint32_t step_id)
{
    return {
        static_cast<uint8_t>((step_id >> 24) & 0xFF),
        static_cast<uint8_t>((step_id >> 16) & 0xFF),
        static_cast<uint8_t>((step_id >>  8) & 0xFF),
        static_cast<uint8_t>( step_id        & 0xFF),
    };
}

/**
 * Create a probe packet: step_id in first 4 bytes, zero-padded to pkt_size.
 * pkt_size is clamped to a minimum of 4.
 *
 * Typical use inside a DoSend function:
 *   g_sendSocket->Send(netrl::CreateProbePacket(step_id, pkt_size));
 */
inline Ptr<Packet> CreateProbePacket(uint32_t step_id, int pkt_size)
{
    auto hdr = EncodeStepId(step_id);
    std::vector<uint8_t> payload(static_cast<std::size_t>(std::max(pkt_size, 4)), 0);
    std::copy(hdr.begin(), hdr.end(), payload.begin());
    return Create<Packet>(payload.data(), static_cast<uint32_t>(payload.size()));
}

// ============================================================================
// Socket helpers
// ============================================================================

/**
 * Create a bound UDP receive socket on node, listening on port.
 * recv_cb is connected as the RecvCallback immediately.
 */
inline Ptr<Socket> CreateUdpRecvSocket(Ptr<Node> node,
                                       uint16_t port,
                                       Callback<void, Ptr<Socket>> recv_cb)
{
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    Ptr<Socket> sock = Socket::CreateSocket(node, tid);
    sock->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    sock->SetRecvCallback(recv_cb);
    return sock;
}

/**
 * Create a connected UDP send socket on sender_node, targeting dest_addr:port.
 */
inline Ptr<Socket> CreateUdpSendSocket(Ptr<Node> sender_node,
                                       Ipv4Address dest_addr,
                                       uint16_t port)
{
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    Ptr<Socket> sock = Socket::CreateSocket(sender_node, tid);
    sock->Connect(InetSocketAddress(dest_addr, port));
    return sock;
}

// ============================================================================
// IPv4 internet stack + addressing
// ============================================================================

/**
 * Install IPv4 internet stack on nodes and assign addresses from the subnet.
 * Returns the interface container (use .GetAddress(i) to retrieve each IP).
 */
inline Ipv4InterfaceContainer InstallInternetStack(NodeContainer& nodes,
                                                   NetDeviceContainer& devices,
                                                   const char* base_ip = "10.1.1.0",
                                                   const char* mask    = "255.255.255.0")
{
    InternetStackHelper internet;
    internet.Install(nodes);
    Ipv4AddressHelper ipv4;
    ipv4.SetBase(base_ip, mask);
    return ipv4.Assign(devices);
}

// ============================================================================
// Static (ConstantPosition) mobility
// ============================================================================

/**
 * Install ConstantPositionMobilityModel on nodes[i] at positions[i].
 * positions must have at least nodes.GetN() entries.
 */
inline void InstallConstantPositions(NodeContainer& nodes,
                                     const std::vector<Vector>& positions)
{
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> pos_alloc = CreateObject<ListPositionAllocator>();
    for (const auto& p : positions)
        pos_alloc->Add(p);
    mobility.SetPositionAllocator(pos_alloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(nodes);
}

// ============================================================================
// 802.11a YANS WiFi helpers
// ============================================================================

/**
 * Aggregates the configured YansWifiPhyHelper and WifiHelper so both can be
 * returned from ConfigureYans80211a() and passed to wifi.Install().
 */
struct YansWifi80211a {
    YansWifiPhyHelper phy;
    WifiHelper        wifi;
};

/**
 * Build the standard 802.11a YANS PHY + STA manager used by ns3_wifi_sim.cc
 * and ns3_wifi_multi_ue_sim.cc.
 *
 * Parameters
 * ----------
 * tx_power_dbm         : TX power in dBm (applied to both TxPowerStart/End)
 * loss_exponent        : Log-distance path-loss exponent
 * reference_loss_db    : Path loss at reference_distance_m (default 46.68 dB,
 *                        which is free-space Friis at 1 m for 5 GHz)
 * reference_distance_m : Reference distance for the loss model (default 1.0 m)
 *
 * Usage
 * -----
 *   auto [phy, wifi] = netrl::ConfigureYans80211a(g_txPowerDbm, g_lossExp);
 *   NetDeviceContainer devs = wifi.Install(phy, mac, g_nodes);
 */
inline YansWifi80211a ConfigureYans80211a(double tx_power_dbm,
                                          double loss_exponent,
                                          double reference_loss_db    = 46.6777,
                                          double reference_distance_m = 1.0)
{
    YansWifiChannelHelper yans_ch;
    yans_ch.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    yans_ch.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                               "Exponent",          DoubleValue(loss_exponent),
                               "ReferenceDistance", DoubleValue(reference_distance_m),
                               "ReferenceLoss",     DoubleValue(reference_loss_db));

    YansWifiPhyHelper phy;
    phy.SetChannel(yans_ch.Create());
    phy.Set("TxPowerStart", DoubleValue(tx_power_dbm));
    phy.Set("TxPowerEnd",   DoubleValue(tx_power_dbm));

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211a);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",    StringValue("OfdmRate54Mbps"),
                                 "ControlMode", StringValue("OfdmRate6Mbps"));

    return {phy, wifi};
}

// ============================================================================
// Warm-up pre-run
// ============================================================================

/**
 * Run the simulator for warmup_ms before real steps begin.
 *
 * This is needed so that:
 *   - 802.11a ad-hoc (IBSS) nodes exchange beacons and merge into the same BSS
 *     (~3 × 102 ms beacon intervals → use min_warmup_ms >= 310 ms)
 *   - 5G EPC nodes complete attach procedures before step 0
 *     (use min_warmup_ms = 500.0 for mmWave / LENA)
 *
 * Returns the warmup duration, which should be stored as g_simStartMs so that
 * all subsequent step-time calculations are anchored correctly.
 *
 * Usage
 * -----
 *   g_simStartMs = netrl::RunWarmup(g_stepMs, 310.0);  // WiFi ad-hoc
 *   g_simStartMs = netrl::RunWarmup(g_stepMs, 500.0);  // mmWave / LENA
 */
inline double RunWarmup(double step_ms, double min_warmup_ms = 310.0)
{
    double warmup_ms = std::max(step_ms * 2.0, min_warmup_ms);
    Simulator::Stop(MilliSeconds(warmup_ms));
    Simulator::Run();
    return warmup_ms;
}

// ============================================================================
// Command loop — full TRANSMIT / FLUSH / RESET / QUIT protocol
// ============================================================================

/**
 * Run the standard stdin/stdout wire protocol loop.
 *
 * This replaces the entire command-dispatch section of main().  It handles
 * step-time arithmetic for both TRANSMIT and FLUSH so callers only need to
 * implement the channel-specific send and reset logic.
 *
 * Template parameters
 * -------------------
 * SendFn   : callable with signature  void(uint32_t step_id, int pkt_size)
 *            Must be a type accepted by ns3::Simulator::Schedule (function
 *            pointer is the safest choice; lambdas work in NS3 >= 3.40).
 * ResetFn  : callable with signature  void()
 *
 * Parameters
 * ----------
 * step_ms          : step duration in milliseconds (read from CLI)
 * sim_start_ms     : reference to the g_simStartMs set by BuildTopology();
 *                    updated after each RESET via reset_fn()
 * arrived_ids      : reference to the global arrivals deque filled by
 *                    the ReceivePacket callback; cleared per FLUSH
 * do_send          : channel-specific send function / lambda
 * reset_fn         : channel-specific reset (Destroy + BuildTopology);
 *                    must update sim_start_ms itself
 * default_pkt_size : fallback when TRANSMIT omits the pkt_size field
 *
 * Wire protocol
 * -------------
 * Python → sim:
 *   TRANSMIT <step_id> [pkt_size]   schedule a probe send; reply OK
 *   FLUSH    <step_id>              advance sim to end of step; reply RECV [id …]
 *   RESET                           rebuild topology from scratch; reply OK
 *   QUIT                            exit loop
 *
 * sim → Python:
 *   READY      emitted by caller before RunCommandLoop (after BuildTopology)
 *   OK         TRANSMIT / RESET acknowledgement
 *   RECV …     FLUSH response (space-separated arrived step_ids, may be empty)
 *   ERROR …    unknown command
 *
 * Note: Simulator::Destroy() is called after the loop exits.
 */
template<typename SendFn, typename ResetFn>
inline void RunCommandLoop(double                   step_ms,
                           double&                  sim_start_ms,
                           std::vector<uint32_t>&   arrived_ids,
                           SendFn                   do_send,
                           ResetFn                  reset_fn,
                           int                      default_pkt_size = 64)
{
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string        cmd;
        iss >> cmd;

        // ---- TRANSMIT <step_id> [pkt_size] ----------------------------------
        if (cmd == "TRANSMIT") {
            uint32_t step_id  = 0;
            int      pkt_size = default_pkt_size;
            iss >> step_id >> pkt_size;

            // Absolute send time: 1% into the step window to avoid scheduling
            // a send at the exact same ns3 time as the previous FLUSH stop.
            double send_abs_ms = sim_start_ms + step_id * step_ms + step_ms * 0.01;
            double delay_ms    = send_abs_ms - Simulator::Now().GetMilliSeconds();

            if (delay_ms > 0.0)
                Simulator::Schedule(MilliSeconds(delay_ms), do_send, step_id, pkt_size);
            else
                Simulator::Schedule(NanoSeconds(1),         do_send, step_id, pkt_size);

            std::cout << "OK" << std::endl;

        // ---- FLUSH <step_id> ------------------------------------------------
        } else if (cmd == "FLUSH") {
            uint32_t step_id = 0;
            iss >> step_id;

            // Advance to the end of this step's window.
            double end_abs_ms = sim_start_ms + (step_id + 1.0) * step_ms;
            double delay_ms   = end_abs_ms - Simulator::Now().GetMilliSeconds();

            arrived_ids.clear();
            if (delay_ms > 0.0) {
                Simulator::Stop(MilliSeconds(delay_ms));
                Simulator::Run();
            }
            // If already past end_abs_ms, nothing new can arrive — no Run().

            std::ostringstream resp;
            resp << "RECV";
            for (uint32_t sid : arrived_ids)
                resp << ' ' << sid;
            std::cout << resp.str() << std::endl;

        // ---- RESET ----------------------------------------------------------
        } else if (cmd == "RESET") {
            reset_fn();   // must call Simulator::Destroy() + BuildTopology()
                          // and update sim_start_ms internally
            std::cout << "OK" << std::endl;

        // ---- QUIT -----------------------------------------------------------
        } else if (cmd == "QUIT") {
            break;

        // ---- Unknown --------------------------------------------------------
        } else {
            std::cout << "ERROR unknown command: " << cmd << std::endl;
        }
    }

    Simulator::Destroy();
}

} // namespace netrl
