/*
 * ns3_lena_sim.cc
 * ===============
 * Persistent 5G-LENA NR simulation (UE -> gNB uplink via EPC).
 *
 * Protocol (stdin/stdout, line-oriented)
 * --------------------------------------
 * Python -> program:
 *   TRANSMIT <step_id> <pkt_size>
 *   FLUSH    <step_id>
 *   RESET
 *   QUIT
 *
 * Program -> Python:
 *   READY
 *   OK
 *   RECV <id1> <id2> ...
 *   ERROR <msg>
 */

#include "ns3/antenna-module.h"
#include "ns3/buildings-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

// ---------------------------------------------------------------------------
// Simulation parameters (set from command line once, reused across RESETs)
// ---------------------------------------------------------------------------
static double g_stepMs = 1.0;
static double g_distanceM = 50.0;
static double g_freqHz = 28.0e9;
static double g_bandwidthHz = 100.0e6;
static double g_ueTxPowerDbm = 23.0;
static double g_gnbTxPowerDbm = 30.0;
static std::string g_scenario = "UMa";
static uint32_t g_numerology = 3;
static bool g_shadowingEnabled = false;
static int g_pktSize = 64;

// ---------------------------------------------------------------------------
// Simulation state (rebuilt on RESET)
// ---------------------------------------------------------------------------
static double g_simStartMs = 0.0;
static std::vector<uint32_t> g_arrivedIds;

static NodeContainer g_gnbNodes;
static NodeContainer g_ueNodes;
static NetDeviceContainer g_gnbDevs;
static NetDeviceContainer g_ueDevs;
static Ptr<Socket> g_sendSocket;
static Ptr<Socket> g_recvSocket;

static Ptr<NrHelper> g_nrHelper;
static Ptr<NrPointToPointEpcHelper> g_epcHelper;
static Ptr<NrChannelHelper> g_channelHelper;
static Ptr<IdealBeamformingHelper> g_beamformingHelper;

static const uint16_t RECV_PORT = 9;

static void
ReceivePacket(Ptr<Socket> socket)
{
    Ptr<Packet> pkt;
    Address from;
    while ((pkt = socket->RecvFrom(from)))
    {
        if (pkt->GetSize() < 4)
            continue;
        uint8_t buf[4];
        pkt->CopyData(buf, 4);
        uint32_t step_id = (uint32_t(buf[0]) << 24) | (uint32_t(buf[1]) << 16) |
                           (uint32_t(buf[2]) << 8) | uint32_t(buf[3]);
        g_arrivedIds.push_back(step_id);
    }
}

static void
DoSend(uint32_t step_id, int pkt_size)
{
    if (!g_sendSocket)
        return;

    uint8_t buf[4];
    buf[0] = static_cast<uint8_t>((step_id >> 24) & 0xFF);
    buf[1] = static_cast<uint8_t>((step_id >> 16) & 0xFF);
    buf[2] = static_cast<uint8_t>((step_id >> 8) & 0xFF);
    buf[3] = static_cast<uint8_t>(step_id & 0xFF);

    std::vector<uint8_t> payload(static_cast<size_t>(std::max(pkt_size, 4)), 0);
    std::copy(buf, buf + 4, payload.begin());

    Ptr<Packet> pkt =
        Create<Packet>(payload.data(), static_cast<uint32_t>(payload.size()));
    g_sendSocket->Send(pkt);
}

static void
BuildTopology()
{
    g_epcHelper = CreateObject<NrPointToPointEpcHelper>();
    g_beamformingHelper = CreateObject<IdealBeamformingHelper>();
    g_nrHelper = CreateObject<NrHelper>();
    g_nrHelper->SetBeamformingHelper(g_beamformingHelper);
    g_nrHelper->SetEpcHelper(g_epcHelper);

    g_beamformingHelper->SetAttribute("BeamformingMethod",
                                      TypeIdValue(DirectPathBeamforming::GetTypeId()));

    g_nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(2));
    g_nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(4));
    g_nrHelper->SetUeAntennaAttribute("AntennaElement",
                                      PointerValue(CreateObject<IsotropicAntennaModel>()));

    g_nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    g_nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(8));
    g_nrHelper->SetGnbAntennaAttribute("AntennaElement",
                                       PointerValue(CreateObject<IsotropicAntennaModel>()));

    BandwidthPartInfoPtrVector allBwps;
    CcBwpCreator ccBwpCreator;
    const uint8_t numCcPerBand = 1;
    CcBwpCreator::SimpleOperationBandConf bandConf(g_freqHz, g_bandwidthHz, numCcPerBand);
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    g_channelHelper = CreateObject<NrChannelHelper>();
    g_channelHelper->ConfigureFactories(g_scenario, "Default", "ThreeGpp");
    g_channelHelper->SetPathlossAttribute("ShadowingEnabled",
                                          BooleanValue(g_shadowingEnabled));
    g_channelHelper->AssignChannelsToBands({band});
    allBwps = CcBwpCreator::GetAllBwps({band});

    g_gnbNodes.Create(1);
    g_ueNodes.Create(1);

    Ptr<ListPositionAllocator> gnbPos = CreateObject<ListPositionAllocator>();
    gnbPos->Add(Vector(0.0, 0.0, 3.0));
    MobilityHelper gnbMobility;
    gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMobility.SetPositionAllocator(gnbPos);
    gnbMobility.Install(g_gnbNodes);
    BuildingsHelper::Install(g_gnbNodes);

    Ptr<ListPositionAllocator> uePos = CreateObject<ListPositionAllocator>();
    uePos->Add(Vector(g_distanceM, 0.0, 1.5));
    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    ueMobility.SetPositionAllocator(uePos);
    ueMobility.Install(g_ueNodes);
    BuildingsHelper::Install(g_ueNodes);

    g_gnbDevs = g_nrHelper->InstallGnbDevice(g_gnbNodes, allBwps);
    g_ueDevs = g_nrHelper->InstallUeDevice(g_ueNodes, allBwps);

    NrHelper::GetGnbPhy(g_gnbDevs.Get(0), 0)->SetAttribute("Numerology",
                                                            UintegerValue(g_numerology));
    NrHelper::GetGnbPhy(g_gnbDevs.Get(0), 0)->SetTxPower(g_gnbTxPowerDbm);
    NrHelper::GetUePhy(g_ueDevs.Get(0), 0)->SetTxPower(g_ueTxPowerDbm);

    for (uint32_t i = 0; i < g_gnbDevs.GetN(); ++i)
    {
        DynamicCast<NrGnbNetDevice>(g_gnbDevs.Get(i))->UpdateConfig();
    }
    for (uint32_t i = 0; i < g_ueDevs.GetN(); ++i)
    {
        DynamicCast<NrUeNetDevice>(g_ueDevs.Get(i))->UpdateConfig();
    }

    auto [remoteHost, _remoteHostIpv4Address] =
        g_epcHelper->SetupRemoteHost("100Gb/s", 2500, Seconds(0.000));

    Ptr<Ipv4> remoteHostIpv4 = remoteHost->GetObject<Ipv4>();
    Ipv4Address remoteHostIpv4Address =
        remoteHostIpv4->GetAddress(1, 0).GetLocal();

    InternetStackHelper internet;
    internet.Install(g_ueNodes);
    Ipv4InterfaceContainer ueIpIfaces =
        g_epcHelper->AssignUeIpv4Address(NetDeviceContainer(g_ueDevs));

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    for (uint32_t j = 0; j < g_ueNodes.GetN(); ++j)
    {
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(g_ueNodes.Get(j)->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(g_epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    g_nrHelper->AttachToClosestGnb(g_ueDevs, g_gnbDevs);

    NrEpsBearer bearer(NrEpsBearer::NGBR_LOW_LAT_EMBB);
    Ptr<NrEpcTft> tft = Create<NrEpcTft>();
    NrEpcTft::PacketFilter pf;
    pf.remotePortStart = RECV_PORT;
    pf.remotePortEnd = RECV_PORT;
    pf.localPortStart = 0;
    pf.localPortEnd = 65535;
    tft->Add(pf);

    for (uint32_t j = 0; j < g_ueDevs.GetN(); ++j)
    {
        g_nrHelper->ActivateDedicatedEpsBearer(g_ueDevs.Get(j), bearer, tft);
    }

    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");

    g_recvSocket = Socket::CreateSocket(remoteHost, tid);
    g_recvSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), RECV_PORT));
    g_recvSocket->SetRecvCallback(MakeCallback(&ReceivePacket));

    g_sendSocket = Socket::CreateSocket(g_ueNodes.Get(0), tid);
    g_sendSocket->Bind(InetSocketAddress(ueIpIfaces.GetAddress(0), 0));
    g_sendSocket->Connect(InetSocketAddress(remoteHostIpv4Address, RECV_PORT));

    const double warmupMs = std::max(500.0, g_stepMs * 2.0);
    Simulator::Stop(MilliSeconds(warmupMs));
    Simulator::Run();
    g_simStartMs = warmupMs;
}

static void
ResetSimulation()
{
    Simulator::Destroy();

    g_nrHelper = nullptr;
    g_epcHelper = nullptr;
    g_channelHelper = nullptr;
    g_beamformingHelper = nullptr;

    g_gnbNodes = NodeContainer();
    g_ueNodes = NodeContainer();
    g_gnbDevs = NetDeviceContainer();
    g_ueDevs = NetDeviceContainer();

    g_sendSocket = nullptr;
    g_recvSocket = nullptr;
    g_arrivedIds.clear();

    BuildTopology();
}

int
main(int argc, char* argv[])
{
    signal(SIGPIPE, SIG_IGN);
    LogComponentDisableAll(LOG_LEVEL_ALL);

    int shadowingInt = 0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("step-ms", "Step duration [ms]", g_stepMs);
    cmd.AddValue("distance", "UE-gNB distance [m]", g_distanceM);
    cmd.AddValue("freq", "Center frequency [Hz]", g_freqHz);
    cmd.AddValue("bandwidth", "Bandwidth [Hz]", g_bandwidthHz);
    cmd.AddValue("ue-tx-power", "UE TX power [dBm]", g_ueTxPowerDbm);
    cmd.AddValue("gnb-tx-power", "gNB TX power [dBm]", g_gnbTxPowerDbm);
    cmd.AddValue("scenario", "3GPP scenario", g_scenario);
    cmd.AddValue("numerology", "NR numerology", g_numerology);
    cmd.AddValue("shadowing", "Enable shadowing (0/1)", shadowingInt);
    cmd.AddValue("pkt-size", "Default packet payload [bytes]", g_pktSize);
    cmd.Parse(argc, argv);

    g_shadowingEnabled = (shadowingInt != 0);

    BuildTopology();
    std::cout << "READY" << std::endl;

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (line.empty())
            continue;

        std::istringstream iss(line);
        std::string command;
        iss >> command;

        if (command == "TRANSMIT")
        {
            uint32_t step_id = 0;
            int pkt_size = g_pktSize;
            iss >> step_id >> pkt_size;

            double sendAbsMs = g_simStartMs + step_id * g_stepMs + g_stepMs * 0.01;
            double nowMs = Simulator::Now().GetMilliSeconds();
            double delayMs = sendAbsMs - nowMs;

            if (delayMs > 0.0)
            {
                Simulator::Schedule(MilliSeconds(delayMs), &DoSend, step_id, pkt_size);
            }
            else
            {
                Simulator::Schedule(NanoSeconds(1), &DoSend, step_id, pkt_size);
            }

            std::cout << "OK" << std::endl;
        }
        else if (command == "FLUSH")
        {
            uint32_t step_id = 0;
            iss >> step_id;

            double endAbsMs = g_simStartMs + (step_id + 1.0) * g_stepMs;
            double nowMs = Simulator::Now().GetMilliSeconds();
            double delayMs = endAbsMs - nowMs;

            g_arrivedIds.clear();
            if (delayMs > 0.0)
            {
                Simulator::Stop(MilliSeconds(delayMs));
                Simulator::Run();
            }

            std::cout << "RECV";
            for (uint32_t sid : g_arrivedIds)
            {
                std::cout << " " << sid;
            }
            std::cout << std::endl;
        }
        else if (command == "RESET")
        {
            ResetSimulation();
            std::cout << "OK" << std::endl;
        }
        else if (command == "QUIT")
        {
            break;
        }
        else
        {
            std::cout << "ERROR unknown command: " << command << std::endl;
        }
    }

    Simulator::Destroy();
    return 0;
}
