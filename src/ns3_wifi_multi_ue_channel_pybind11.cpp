/**
 * ns3_wifi_multi_ue_channel_pybind11.cpp
 *
 * Multi-UE WiFi simulator using pybind11 for Python integration.
 * Replaces the subprocess-based ns3_wifi_multi_ue_sim protocol with
 * direct in-process C++ calls.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ns3/core-module.h>
#include <ns3/internet-module.h>
#include <ns3/mobility-module.h>
#include <ns3/network-module.h>
#include <ns3/wifi-module.h>

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <sstream>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace ns3;

class NS3WiFiMultiUEChannel {
  public:
    NS3WiFiMultiUEChannel(int n_ues,
                          const std::vector<double>& distances_m,
                          double step_duration_ms = 1.0,
                          double tx_power_dbm = 20.0,
                          double loss_exponent = 3.0,
                          int max_retries = 7,
                          int packet_size_bytes = 64)
        : n_ues_(n_ues),
          distances_m_(distances_m),
          step_duration_ms_(step_duration_ms),
          tx_power_dbm_(tx_power_dbm),
          loss_exponent_(loss_exponent),
          max_retries_(max_retries),
          packet_size_bytes_(packet_size_bytes)
    {
        signal(SIGPIPE, SIG_IGN);
        LogComponentDisableAll(LOG_LEVEL_ALL);
        ParseDistances();
        BuildTopology();
    }

    ~NS3WiFiMultiUEChannel() { Simulator::Destroy(); }

    void transmit(int ue_id, int step_id, int packet_size = -1)
    {
        if (packet_size < 0) {
            packet_size = packet_size_bytes_;
        }
        if (ue_id < 0 || ue_id >= n_ues_) {
            throw std::runtime_error("ue_id out of range");
        }

        const uint32_t uid = static_cast<uint32_t>(ue_id);
        const uint32_t sid = static_cast<uint32_t>(step_id);

        double offset_frac = 0.0; // TODO: Could randomize this per UE for more realism
        double send_abs_ms = sim_start_ms_ + sid * step_duration_ms_ + step_duration_ms_ * offset_frac;
        double now_ms = Simulator::Now().GetMilliSeconds();
        double delay_ms = send_abs_ms - now_ms;

        if (delay_ms > 0.0) {
            Simulator::Schedule(MilliSeconds(delay_ms),
                                &NS3WiFiMultiUEChannel::DoSend,
                                this,
                                uid,
                                sid,
                                packet_size);
        } else {
            Simulator::Schedule(NanoSeconds(1),
                                &NS3WiFiMultiUEChannel::DoSend,
                                this,
                                uid,
                                sid,
                                packet_size);
        }
    }

    std::vector<std::pair<int, int>> flush(int step_id)
    {
        double end_abs_ms = sim_start_ms_ + (step_id + 1.0) * step_duration_ms_;
        double now_ms = Simulator::Now().GetMilliSeconds();
        double delay_ms = end_abs_ms - now_ms;

        arrived_pairs_.clear();

        if (delay_ms > 0.0) {
            Simulator::Stop(MilliSeconds(delay_ms));
            Simulator::Run();
        }

        std::vector<std::pair<int, int>> out;
        out.reserve(arrived_pairs_.size());
        for (const auto& p : arrived_pairs_) {
            out.push_back({static_cast<int>(p.first), static_cast<int>(p.second)});
        }
        return out;
    }

    void reset()
    {
        Simulator::Destroy();
        nodes_ = NodeContainer();
        send_sockets_.clear();
        recv_socket_ = nullptr;
        arrived_pairs_.clear();
        ParseDistances();
        BuildTopology();
    }

    py::dict get_channel_info() const
    {
        py::dict d;
        d["state"] = "NS3_WIFI_MULTI_UE";
        d["n_ues"] = n_ues_;
        d["step_duration_ms"] = step_duration_ms_;
        d["tx_power_dbm"] = tx_power_dbm_;
        d["loss_exponent"] = loss_exponent_;
        d["max_retries"] = max_retries_;
        d["packet_size_bytes"] = packet_size_bytes_;
        d["pending_arrivals"] = static_cast<int>(arrived_pairs_.size());
        return d;
    }

  private:
    void ParseDistances()
    {
        if (distances_m_.empty()) {
            distances_m_.push_back(10.0);
        }
        while (static_cast<int>(distances_m_.size()) < n_ues_) {
            distances_m_.push_back(distances_m_.back());
        }
    }

    void ReceivePacket(Ptr<Socket> socket)
    {
        Ptr<Packet> pkt;
        Address from;
        while ((pkt = socket->RecvFrom(from)) != nullptr) {
            if (pkt->GetSize() >= 8) {
                uint8_t buf[8];
                pkt->CopyData(buf, 8);
                uint32_t ue_id = (static_cast<uint32_t>(buf[0]) << 24) |
                                 (static_cast<uint32_t>(buf[1]) << 16) |
                                 (static_cast<uint32_t>(buf[2]) << 8) |
                                 static_cast<uint32_t>(buf[3]);
                uint32_t step_id = (static_cast<uint32_t>(buf[4]) << 24) |
                                   (static_cast<uint32_t>(buf[5]) << 16) |
                                   (static_cast<uint32_t>(buf[6]) << 8) |
                                   static_cast<uint32_t>(buf[7]);
                arrived_pairs_.push_back({ue_id, step_id});
            }
        }
    }

    void DoSend(uint32_t ue_id, uint32_t step_id, int pkt_size)
    {
        uint8_t hdr[8];
        hdr[0] = static_cast<uint8_t>((ue_id >> 24) & 0xFF);
        hdr[1] = static_cast<uint8_t>((ue_id >> 16) & 0xFF);
        hdr[2] = static_cast<uint8_t>((ue_id >> 8) & 0xFF);
        hdr[3] = static_cast<uint8_t>(ue_id & 0xFF);
        hdr[4] = static_cast<uint8_t>((step_id >> 24) & 0xFF);
        hdr[5] = static_cast<uint8_t>((step_id >> 16) & 0xFF);
        hdr[6] = static_cast<uint8_t>((step_id >> 8) & 0xFF);
        hdr[7] = static_cast<uint8_t>(step_id & 0xFF);

        std::vector<uint8_t> payload(std::max(pkt_size, 8), 0);
        std::copy(hdr, hdr + 8, payload.begin());

        Ptr<Packet> pkt = Create<Packet>(payload.data(), static_cast<uint32_t>(payload.size()));
        if (ue_id < send_sockets_.size()) {
            send_sockets_[ue_id]->Send(pkt);
        }
    }

    void BuildTopology()
    {
        nodes_.Create(static_cast<uint32_t>(n_ues_ + 1));

        YansWifiChannelHelper channel_helper;
        channel_helper.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
        channel_helper.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                                          "Exponent", DoubleValue(loss_exponent_),
                                          "ReferenceDistance", DoubleValue(1.0),
                                          "ReferenceLoss", DoubleValue(46.6777));

        YansWifiPhyHelper phy_helper;
        phy_helper.SetChannel(channel_helper.Create());
        phy_helper.Set("TxPowerStart", DoubleValue(tx_power_dbm_));
        phy_helper.Set("TxPowerEnd", DoubleValue(tx_power_dbm_));

        WifiHelper wifi_helper;
        wifi_helper.SetStandard(WIFI_STANDARD_80211a);
        wifi_helper.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                             "DataMode", StringValue("OfdmRate54Mbps"),
                                             "ControlMode", StringValue("OfdmRate6Mbps"));

        Ssid ssid = Ssid("netrl-bss");
        WifiMacHelper mac_ap;
        mac_ap.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
        NetDeviceContainer ap_device = wifi_helper.Install(phy_helper, mac_ap, nodes_.Get(0));

        WifiMacHelper mac_sta;
        mac_sta.SetType("ns3::StaWifiMac",
                        "Ssid", SsidValue(ssid),
                        "ActiveProbing", BooleanValue(true));

        NodeContainer sta_nodes;
        for (int i = 0; i < n_ues_; ++i) {
            sta_nodes.Add(nodes_.Get(static_cast<uint32_t>(i + 1)));
        }
        NetDeviceContainer sta_devices = wifi_helper.Install(phy_helper, mac_sta, sta_nodes);

        Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/FrameRetryLimit",
                    UintegerValue(static_cast<uint32_t>(max_retries_)));

        MobilityHelper mobility;
        Ptr<ListPositionAllocator> pos_alloc = CreateObject<ListPositionAllocator>();
        pos_alloc->Add(Vector(0.0, 0.0, 0.0));
        for (int i = 0; i < n_ues_; ++i) {
            const double d = distances_m_[static_cast<size_t>(i)];
            pos_alloc->Add(Vector(d, 0.0, 0.0));
        }
        mobility.SetPositionAllocator(pos_alloc);
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mobility.Install(nodes_);

        InternetStackHelper internet;
        internet.Install(nodes_);

        NetDeviceContainer all_devices;
        all_devices.Add(ap_device);
        all_devices.Add(sta_devices);
        Ipv4AddressHelper ipv4;
        ipv4.SetBase("10.1.1.0", "255.255.255.0");
        Ipv4InterfaceContainer iface = ipv4.Assign(all_devices);

        TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");

        recv_socket_ = Socket::CreateSocket(nodes_.Get(0), tid);
        recv_socket_->Bind(InetSocketAddress(Ipv4Address::GetAny(), 9));
        recv_socket_->SetRecvCallback(MakeCallback(&NS3WiFiMultiUEChannel::ReceivePacket, this));

        Ipv4Address ap_addr = iface.GetAddress(0);
        send_sockets_.resize(static_cast<size_t>(n_ues_));
        for (int i = 0; i < n_ues_; ++i) {
            send_sockets_[static_cast<size_t>(i)] =
                Socket::CreateSocket(nodes_.Get(static_cast<uint32_t>(i + 1)), tid);
            send_sockets_[static_cast<size_t>(i)]->Connect(InetSocketAddress(ap_addr, 9));
        }

        const double warmup_ms = 500.0;
        Simulator::Stop(MilliSeconds(warmup_ms));
        Simulator::Run();
        sim_start_ms_ = warmup_ms;
    }

  public:
    int n_ues_;
    std::vector<double> distances_m_;
    double step_duration_ms_;
    double tx_power_dbm_;
    double loss_exponent_;
    int max_retries_;
    int packet_size_bytes_;

    double sim_start_ms_ = 0.0;
    NodeContainer nodes_;
    std::vector<Ptr<Socket>> send_sockets_;
    Ptr<Socket> recv_socket_;
    std::vector<std::pair<uint32_t, uint32_t>> arrived_pairs_;
};

PYBIND11_MODULE(_netrl_multi_ue_ext, m)
{
    m.doc() = "NetRL multi-UE NS3 channel extension";

    py::class_<NS3WiFiMultiUEChannel>(m, "NS3WiFiMultiUEChannel")
        .def(py::init<int, const std::vector<double>&, double, double, double, int, int>(),
             py::arg("n_ues"),
             py::arg("distances_m"),
             py::arg("step_duration_ms") = 1.0,
             py::arg("tx_power_dbm") = 20.0,
             py::arg("loss_exponent") = 3.0,
             py::arg("max_retries") = 7,
             py::arg("packet_size_bytes") = 64)
        .def("transmit",
             &NS3WiFiMultiUEChannel::transmit,
             py::arg("ue_id"),
             py::arg("step_id"),
             py::arg("packet_size") = -1)
        .def("flush", &NS3WiFiMultiUEChannel::flush, py::arg("step_id"))
        .def("reset", &NS3WiFiMultiUEChannel::reset)
        .def("get_channel_info", &NS3WiFiMultiUEChannel::get_channel_info);
}
