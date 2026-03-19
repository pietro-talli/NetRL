/**
 * ns3_wifi_adhoc_sim_example.cc
 *
 * A minimal 802.11a ad-hoc WiFi channel simulator written using the
 * ns3_channel_utils.h helpers.  Compare to the original ns3_wifi_sim.cc
 * (347 lines) — this version is ~110 lines and contains only the
 * channel-specific logic.
 *
 * This file is a REFERENCE EXAMPLE showing how to write a new channel.
 * It is NOT a replacement for ns3_wifi_sim.cc; both can coexist.
 *
 * How to build (same flags as ns3_wifi_sim.cc):
 *   g++ -std=c++20 -O2 ns3_wifi_adhoc_sim_example.cc \
 *       -I$(NS3_INC) -L$(NS3_LIB) -lns3.*-core ... -o ns3_wifi_adhoc_example
 */

// --- NS3 modules first, then the shared utilities --------------------------
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include "ns3_channel_utils.h"   // <-- shared utilities

#include <csignal>
#include <iostream>
#include <vector>

using namespace ns3;

// ---------------------------------------------------------------------------
// Channel parameters (set once from CLI)
// ---------------------------------------------------------------------------
static double g_stepMs    = 1.0;
static double g_distanceM = 10.0;
static double g_txPowerDbm = 20.0;
static double g_lossExp   = 3.0;
static int    g_maxRetries = 7;
static int    g_pktSize   = 64;

// ---------------------------------------------------------------------------
// Simulation state (rebuilt on RESET)
// ---------------------------------------------------------------------------
static NodeContainer         g_nodes;
static Ptr<Socket>           g_sendSocket;
static Ptr<Socket>           g_recvSocket;
static std::vector<uint32_t> g_arrivedIds;
static double                g_simStartMs = 0.0;

// ---------------------------------------------------------------------------
// Receive callback — only channel-unique line: where to store the step_id
// ---------------------------------------------------------------------------
static void ReceivePacket(Ptr<Socket> sock)
{
    Ptr<Packet> pkt;
    Address from;
    while ((pkt = sock->RecvFrom(from))) {
        if (pkt->GetSize() < 4) continue;
        uint8_t buf[4];
        pkt->CopyData(buf, 4);
        g_arrivedIds.push_back(netrl::DecodeStepId(buf));   // utility
    }
}

// ---------------------------------------------------------------------------
// Scheduled send — only channel-unique line: which socket to use
// ---------------------------------------------------------------------------
static void DoSend(uint32_t step_id, int pkt_size)
{
    g_sendSocket->Send(netrl::CreateProbePacket(step_id, pkt_size));  // utility
}

// ---------------------------------------------------------------------------
// BuildTopology — channel-specific; everything else delegated to utilities
// ---------------------------------------------------------------------------
static void BuildTopology()
{
    g_nodes.Create(2);  // node 0 = STA, node 1 = AP

    // WiFi PHY + rate manager (utility)
    auto [phy, wifi] = netrl::ConfigureYans80211a(g_txPowerDbm, g_lossExp);

    // MAC: ad-hoc (no association delay)
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devs = wifi.Install(phy, mac, g_nodes);

    // Set MAC retry limit
    Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/FrameRetryLimit",
                UintegerValue(static_cast<uint32_t>(g_maxRetries)));

    // Static positions (utility)
    netrl::InstallConstantPositions(g_nodes, {
        Vector(0.0,         0.0, 0.0),   // STA
        Vector(g_distanceM, 0.0, 0.0),   // AP
    });

    // Internet stack + addressing (utility)
    Ipv4InterfaceContainer iface = netrl::InstallInternetStack(g_nodes, devs);

    // Sockets (utility)
    g_recvSocket = netrl::CreateUdpRecvSocket(g_nodes.Get(1), 9,
                                              MakeCallback(&ReceivePacket));
    g_sendSocket = netrl::CreateUdpSendSocket(g_nodes.Get(0),
                                              iface.GetAddress(1), 9);

    // IBSS warm-up (utility); returns ms → becomes step-0 anchor
    g_simStartMs = netrl::RunWarmup(g_stepMs, 310.0);
}

// ---------------------------------------------------------------------------
// ResetSimulation — destroy everything and rebuild from scratch
// ---------------------------------------------------------------------------
static void ResetSimulation()
{
    Simulator::Destroy();
    g_nodes       = NodeContainer();
    g_sendSocket  = nullptr;
    g_recvSocket  = nullptr;
    g_arrivedIds.clear();
    BuildTopology();   // also resets g_simStartMs
}

// ---------------------------------------------------------------------------
// main — init + parse args + build topology + hand off to command loop
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    netrl::InitNS3();   // utility: SIGPIPE + silence logging

    CommandLine cmd(__FILE__);
    cmd.AddValue("step-ms",  "Step duration in milliseconds",    g_stepMs);
    cmd.AddValue("distance", "STA–AP distance in metres",        g_distanceM);
    cmd.AddValue("tx-power", "TX power in dBm",                  g_txPowerDbm);
    cmd.AddValue("loss-exp", "Log-distance path-loss exponent",  g_lossExp);
    cmd.AddValue("retries",  "Max MAC retry count",              g_maxRetries);
    cmd.AddValue("pkt-size", "Probe packet payload bytes",       g_pktSize);
    cmd.Parse(argc, argv);

    BuildTopology();
    std::cout << "READY" << std::endl;

    // Entire command loop handled by the utility
    netrl::RunCommandLoop(g_stepMs, g_simStartMs, g_arrivedIds,
                          &DoSend, &ResetSimulation, g_pktSize);
    return 0;
}
