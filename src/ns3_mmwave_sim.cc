/*
 * ns3_mmwave_sim.cc
 * =================
 * Persistent 5G mmWave channel simulation (UE → eNB uplink via EPC).
 *
 * Controlled by NetRL over stdin/stdout with the same line protocol used
 * by ns3_wifi_sim:
 *
 * Protocol (stdin / stdout, line-oriented)
 * -----------------------------------------
 * Python → program:
 *   TRANSMIT <step_id> <pkt_size>  schedule a UDP probe packet for step step_id
 *   FLUSH    <step_id>             advance sim to end of step_id, report arrivals
 *   RESET                          destroy & rebuild simulation (sim time → 0)
 *   QUIT                           graceful exit
 *
 * Program → Python:
 *   READY                  emitted once at startup (sim is ready)
 *   OK                     response to TRANSMIT / RESET
 *   RECV <id1> <id2> ...   response to FLUSH — step_ids that arrived
 *   ERROR <msg>            unexpected condition
 *
 * Topology
 * --------
 *   UE (agent) ── mmWave PHY/MAC ── eNB ── P2P 100Gbps ── PGW ── Remote Host
 *
 *   UE sends UDP packets to Remote Host (uplink direction).
 *   Remote Host's recv socket fires ReceivePacket() to log arrived step_ids.
 *   UseIdealRrc = true: RRC/bearer setup is instant (no OTA delay).
 *
 * Command-line arguments
 * ----------------------
 *   --step-ms        step duration [ms]                (default: 1.0)
 *   --distance       UE-eNB distance [m]               (default: 50.0)
 *   --freq           centre frequency [GHz]            (default: 28.0)
 *   --bandwidth      bandwidth [GHz]                   (default: 0.2)
 *   --tx-power       UE transmit power [dBm]           (default: 23.0)
 *   --enb-tx-power   eNB transmit power [dBm]          (default: 30.0)
 *   --noise-fig      UE noise figure [dB]              (default: 9.0)
 *   --enb-noise-fig  eNB noise figure [dB]             (default: 5.0)
 *   --scenario       3GPP scenario string              (default: UMa)
 *                    "RMa","UMa","UMi-StreetCanyon",
 *                    "InH-OfficeMixed","InH-OfficeOpen"
 *   --harq           HARQ enabled  0/1                 (default: 1)
 *   --rlc-am         RLC-AM enabled 0/1                (default: 0)
 *   --pkt-size       default packet size [bytes]       (default: 64)
 */

#include "ns3/buildings-helper.h"
#include "ns3/buildings-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mmwave-helper.h"
#include "ns3/mmwave-point-to-point-epc-helper.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-helper.h"

#include <csignal>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;
using namespace mmwave;

// ---------------------------------------------------------------------------
// Simulation parameters (set from command line once, reused across RESETs)
// ---------------------------------------------------------------------------
static double      g_stepMs          = 1.0;
static double      g_distanceM       = 50.0;
static double      g_freqHz          = 28.0e9;
static double      g_bandwidthHz     = 200.0e6;
static double      g_txPowerDbm      = 23.0;
static double      g_enbTxPowerDbm   = 30.0;
static double      g_noiseFigureDb   = 9.0;
static double      g_enbNoiseFigureDb = 5.0;
static std::string g_scenario        = "UMa";
static bool        g_harq            = true;
static bool        g_rlcAm           = false;
static int         g_pktSize         = 64;

// ---------------------------------------------------------------------------
// Simulation state (rebuilt on RESET)
// ---------------------------------------------------------------------------
static double                  g_simStartMs  = 0.0;
static std::vector<uint32_t>   g_arrivedIds;

static NodeContainer  g_enbNodes;
static NodeContainer  g_ueNodes;
static NodeContainer  g_remoteHostContainer;
static Ptr<Socket>    g_sendSocket;     // UDP socket on UE node
static Ptr<Socket>    g_recvSocket;     // UDP socket on remote host
static Ipv4Address    g_remoteHostAddr;

// Keep helpers alive so that the SGW/PGW node and all EPC devices are not
// prematurely destroyed when BuildTopology() returns.
static Ptr<MmWaveHelper>                g_mmwaveHelper;
static Ptr<MmWavePointToPointEpcHelper> g_epcHelper;

static const uint16_t RECV_PORT = 9;

// ---------------------------------------------------------------------------
// Packet receive callback — called by ns-3 when a UDP packet reaches the
// remote host.  Extracts the 4-byte big-endian step_id from the payload.
// ---------------------------------------------------------------------------
static void
ReceivePacket(Ptr<Socket> socket)
{
    Ptr<Packet> pkt;
    Address     from;
    while ((pkt = socket->RecvFrom(from)))
    {
        if (pkt->GetSize() < 4)
            continue;
        uint8_t buf[4];
        pkt->CopyData(buf, 4);
        uint32_t step_id = (uint32_t(buf[0]) << 24)
                         | (uint32_t(buf[1]) << 16)
                         | (uint32_t(buf[2]) <<  8)
                         |  uint32_t(buf[3]);
        g_arrivedIds.push_back(step_id);
    }
}

// ---------------------------------------------------------------------------
// Scheduled send — fired by ns-3 at the right simulation time
// ---------------------------------------------------------------------------
static void
DoSend(uint32_t step_id, int pkt_size)
{
    if (!g_sendSocket)
        return;

    // Build 4-byte big-endian step_id header
    uint8_t buf[4];
    buf[0] = static_cast<uint8_t>((step_id >> 24) & 0xFF);
    buf[1] = static_cast<uint8_t>((step_id >> 16) & 0xFF);
    buf[2] = static_cast<uint8_t>((step_id >>  8) & 0xFF);
    buf[3] = static_cast<uint8_t>( step_id        & 0xFF);

    // Pad to requested size (at least 4 bytes)
    std::vector<uint8_t> payload(std::max(pkt_size, 4), 0);
    std::copy(buf, buf + 4, payload.begin());

    Ptr<Packet> pkt = Create<Packet>(payload.data(),
                                     static_cast<uint32_t>(payload.size()));
    g_sendSocket->Send(pkt);
}

// ---------------------------------------------------------------------------
// Build (or rebuild) the 5G mmWave topology
// ---------------------------------------------------------------------------
static void
BuildTopology()
{
    // ---- Apply Config::SetDefault BEFORE objects are created ---------------
    // These calls configures the attribute defaults for all objects created
    // afterwards within this simulation run.

    Config::SetDefault("ns3::MmWaveHelper::HarqEnabled",
                       BooleanValue(g_harq));
    Config::SetDefault("ns3::MmWaveHelper::RlcAmEnabled",
                       BooleanValue(g_rlcAm));
    Config::SetDefault("ns3::MmWaveHelper::UseIdealRrc",
                       BooleanValue(true));

    Config::SetDefault("ns3::MmWaveFlexTtiMacScheduler::HarqEnabled",
                       BooleanValue(g_harq));

    // PHY/MAC numerology
    Config::SetDefault("ns3::MmWavePhyMacCommon::CenterFreq",
                       DoubleValue(g_freqHz));
    Config::SetDefault("ns3::MmWavePhyMacCommon::Bandwidth",
                       DoubleValue(g_bandwidthHz));
    Config::SetDefault("ns3::MmWavePhyMacCommon::TbDecodeLatency",
                       UintegerValue(200));

    // TX power and noise figure
    Config::SetDefault("ns3::MmWaveEnbPhy::TxPower",
                       DoubleValue(g_enbTxPowerDbm));
    Config::SetDefault("ns3::MmWaveUePhy::TxPower",
                       DoubleValue(g_txPowerDbm));
    Config::SetDefault("ns3::MmWaveEnbPhy::NoiseFigure",
                       DoubleValue(g_enbNoiseFigureDb));
    Config::SetDefault("ns3::MmWaveUePhy::NoiseFigure",
                       DoubleValue(g_noiseFigureDb));

    // 3GPP channel model: freeze during the episode (very long period)
    Config::SetDefault("ns3::ThreeGppChannelModel::Scenario",
                       StringValue(g_scenario));
    Config::SetDefault("ns3::ThreeGppChannelModel::UpdatePeriod",
                       TimeValue(MilliSeconds(1e6)));
    Config::SetDefault("ns3::ThreeGppChannelModel::Blockage",
                       BooleanValue(false));

    // EPC / RRC timing — minimise connection-setup latency
    Config::SetDefault("ns3::MmWavePointToPointEpcHelper::S1apLinkDelay",
                       TimeValue(Seconds(0)));
    Config::SetDefault("ns3::LteEnbRrc::SystemInformationPeriodicity",
                       TimeValue(MilliSeconds(5.0)));
    Config::SetDefault("ns3::LteEnbRrc::SrsPeriodicity",
                       UintegerValue(20));
    Config::SetDefault("ns3::LteEnbRrc::FirstSibTime",
                       UintegerValue(2));
    Config::SetDefault("ns3::LteRlcAm::ReportBufferStatusTimer",
                       TimeValue(MicroSeconds(100.0)));
    Config::SetDefault("ns3::LteRlcUmLowLat::ReportBufferStatusTimer",
                       TimeValue(MicroSeconds(100.0)));

    // ---- Helpers -----------------------------------------------------------
    // Stored as globals so they outlive BuildTopology() and keep the
    // SGW/PGW node alive for the duration of the simulation run.
    g_mmwaveHelper = CreateObject<MmWaveHelper>();
    g_epcHelper    = CreateObject<MmWavePointToPointEpcHelper>();
    g_mmwaveHelper->SetEpcHelper(g_epcHelper);
    g_mmwaveHelper->SetHarqEnabled(g_harq);

    // ---- Remote host connected to PGW via fast P2P link -------------------
    Ptr<Node> pgw = g_epcHelper->GetPgwNode();

    g_remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = g_remoteHostContainer.Get(0);

    InternetStackHelper internet;
    internet.Install(g_remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu",      UintegerValue(1500));
    p2ph.SetChannelAttribute("Delay",   TimeValue(MicroSeconds(1.0)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIfaces = ipv4h.Assign(internetDevices);
    g_remoteHostAddr = internetIfaces.GetAddress(1);  // 1.0.0.2

    // Route UE subnet (7.0.0.0/8) back from remote host toward PGW
    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteStaticRouting->AddNetworkRouteTo(
        Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    // ---- eNB and UE nodes -------------------------------------------------
    g_enbNodes.Create(1);
    g_ueNodes.Create(1);

    // eNB: fixed at origin, elevated 3 m (typical macro height)
    Ptr<ListPositionAllocator> enbPos = CreateObject<ListPositionAllocator>();
    enbPos->Add(Vector(0.0, 0.0, 3.0));
    MobilityHelper enbMobility;
    enbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbMobility.SetPositionAllocator(enbPos);
    enbMobility.Install(g_enbNodes);
    BuildingsHelper::Install(g_enbNodes);

    // UE: fixed at (distance, 0, 1.5 m) — typical pedestrian height
    Ptr<ListPositionAllocator> uePos = CreateObject<ListPositionAllocator>();
    uePos->Add(Vector(g_distanceM, 0.0, 1.5));
    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    ueMobility.SetPositionAllocator(uePos);
    ueMobility.Install(g_ueNodes);
    BuildingsHelper::Install(g_ueNodes);

    // ---- Install mmWave devices -------------------------------------------
    NetDeviceContainer enbDevs = g_mmwaveHelper->InstallEnbDevice(g_enbNodes);
    NetDeviceContainer ueDevs  = g_mmwaveHelper->InstallUeDevice(g_ueNodes);

    // ---- Internet stack on UE and IP assignment via EPC -------------------
    internet.Install(g_ueNodes);
    Ipv4InterfaceContainer ueIpIface =
        g_epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));

    // Default route on UE toward EPC gateway
    Ptr<Ipv4StaticRouting> ueStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(g_ueNodes.Get(0)->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(
        g_epcHelper->GetUeDefaultGatewayAddress(), 1);

    // ---- Attach UE to eNB (with UseIdealRrc the bearer is set up fast) ----
    g_mmwaveHelper->AttachToClosestEnb(ueDevs, enbDevs);

    // ---- Sockets ----------------------------------------------------------
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");

    // Receive socket on remote host
    g_recvSocket = Socket::CreateSocket(remoteHost, tid);
    g_recvSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), RECV_PORT));
    g_recvSocket->SetRecvCallback(MakeCallback(&ReceivePacket));

    // Send socket on UE, pre-connected to remote host
    g_sendSocket = Socket::CreateSocket(g_ueNodes.Get(0), tid);
    g_sendSocket->Connect(InetSocketAddress(g_remoteHostAddr, RECV_PORT));

    // ---- Warm-up -----------------------------------------------------------
    // Run the simulator long enough for:
    //   1. eNB to broadcast SIB (FirstSibTime = 2 ms)
    //   2. UE to attach (UseIdealRrc → nearly instant)
    //   3. Default bearer to be established
    //   4. Any initial scheduler allocations to complete
    // 500 ms is conservative; with UseIdealRrc attachment finishes < 50 ms.
    const double warmupMs = std::max(500.0, g_stepMs * 2.0);
    Simulator::Stop(MilliSeconds(warmupMs));
    Simulator::Run();
    g_simStartMs = warmupMs;
}

// ---------------------------------------------------------------------------
// Full reset — tear down every ns-3 object and rebuild from scratch
// ---------------------------------------------------------------------------
static void
ResetSimulation()
{
    Simulator::Destroy();
    g_mmwaveHelper        = nullptr;
    g_epcHelper           = nullptr;
    g_enbNodes            = NodeContainer();
    g_ueNodes             = NodeContainer();
    g_remoteHostContainer = NodeContainer();
    g_sendSocket          = nullptr;
    g_recvSocket          = nullptr;
    g_arrivedIds.clear();
    BuildTopology();
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
    signal(SIGPIPE, SIG_IGN);

    // Silence all ns-3 log output so stdout stays clean for the protocol
    LogComponentDisableAll(LOG_LEVEL_ALL);

    // ---- Parse command-line arguments before any object creation ----------
    int  harqInt  = 1;
    int  rlcAmInt = 0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("step-ms",       "Step duration [ms]",              g_stepMs);
    cmd.AddValue("distance",      "UE-eNB distance [m]",             g_distanceM);
    cmd.AddValue("freq",          "Centre frequency [GHz] (×1e9)",   g_freqHz);
    cmd.AddValue("bandwidth",     "Bandwidth [GHz] (×1e9)",          g_bandwidthHz);
    cmd.AddValue("tx-power",      "UE TX power [dBm]",               g_txPowerDbm);
    cmd.AddValue("enb-tx-power",  "eNB TX power [dBm]",              g_enbTxPowerDbm);
    cmd.AddValue("noise-fig",     "UE noise figure [dB]",            g_noiseFigureDb);
    cmd.AddValue("enb-noise-fig", "eNB noise figure [dB]",           g_enbNoiseFigureDb);
    cmd.AddValue("scenario",      "3GPP scenario string",            g_scenario);
    cmd.AddValue("harq",          "HARQ enabled (0/1)",              harqInt);
    cmd.AddValue("rlc-am",        "RLC-AM enabled (0/1)",            rlcAmInt);
    cmd.AddValue("pkt-size",      "Default packet payload [bytes]",  g_pktSize);
    cmd.Parse(argc, argv);

    g_harq  = (harqInt  != 0);
    g_rlcAm = (rlcAmInt != 0);

    // ---- Build initial topology and run warm-up ---------------------------
    BuildTopology();

    // Signal to Python that the simulation is ready
    std::cout << "READY" << std::endl;

    // ---- Command loop ------------------------------------------------------
    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string        command;
        iss >> command;

        // ---- TRANSMIT <step_id> <pkt_size> ---------------------------------
        if (command == "TRANSMIT")
        {
            uint32_t step_id  = 0;
            int      pkt_size = g_pktSize;
            iss >> step_id >> pkt_size;

            // Schedule the send 1% into the step window
            double sendAbsMs = g_simStartMs + step_id * g_stepMs
                             + g_stepMs * 0.01;
            double nowMs     = Simulator::Now().GetMilliSeconds();
            double delayMs   = sendAbsMs - nowMs;

            if (delayMs > 0.0)
                Simulator::Schedule(MilliSeconds(delayMs),
                                    &DoSend, step_id, pkt_size);
            else
                Simulator::Schedule(NanoSeconds(1),
                                    &DoSend, step_id, pkt_size);

            std::cout << "OK" << std::endl;

        // ---- FLUSH <step_id> -----------------------------------------------
        } else if (command == "FLUSH") {
            uint32_t step_id = 0;
            iss >> step_id;

            double endAbsMs = g_simStartMs + (step_id + 1.0) * g_stepMs;
            double nowMs    = Simulator::Now().GetMilliSeconds();
            double delayMs  = endAbsMs - nowMs;

            g_arrivedIds.clear();
            if (delayMs > 0.0)
            {
                Simulator::Stop(MilliSeconds(delayMs));
                Simulator::Run();
            }

            // Emit arrived step_ids
            std::cout << "RECV";
            for (uint32_t sid : g_arrivedIds)
                std::cout << " " << sid;
            std::cout << std::endl;

        // ---- RESET ---------------------------------------------------------
        } else if (command == "RESET") {
            ResetSimulation();
            std::cout << "OK" << std::endl;

        // ---- QUIT ----------------------------------------------------------
        } else if (command == "QUIT") {
            break;

        } else {
            std::cout << "ERROR unknown command: " << command << std::endl;
        }
    }

    Simulator::Destroy();
    return 0;
}
