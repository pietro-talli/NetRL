/**
 * ns3_wifi_multi_ue_channel_pybind11.cpp
 *
 * Multi-UE WiFi channel simulator using pybind11 for Python integration.
 * Replaces the subprocess-based ns3_wifi_multi_ue_sim with direct in-process
 * C++ calls.
 *
 * Shared NS3 boilerplate (YANS config, mobility, internet stack, sockets,
 * packet encoding, warmup) lives in ns3_channel_utils.h.
 *
 * Packet format (8 bytes): [ue_id: 4 B big-endian][step_id: 4 B big-endian]
 * Both fields are encoded/decoded with netrl::EncodeStepId / DecodeStepId.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ns3/core-module.h>
#include <ns3/internet-module.h>
#include <ns3/mobility-module.h>
#include <ns3/network-module.h>
#include <ns3/wifi-module.h>

#include "ns3_channel_utils.h"

#include <cstdint>
#include <sstream>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace ns3;

// ============================================================================
// Multi-UE WiFi Channel Implementation
// ============================================================================

class NS3WiFiMultiUEChannel {
  public:
    NS3WiFiMultiUEChannel(int                        n_ues,
                          const std::vector<double>& distances_m,
                          double  step_duration_ms  = 1.0,
                          double  tx_power_dbm      = 20.0,
                          double  loss_exponent     = 3.0,
                          int     max_retries       = 7,
                          int     packet_size_bytes = 64)
        : n_ues_(n_ues),
          distances_m_(distances_m),
          step_duration_ms_(step_duration_ms),
          tx_power_dbm_(tx_power_dbm),
          loss_exponent_(loss_exponent),
          max_retries_(max_retries),
          packet_size_bytes_(packet_size_bytes)
    {
        netrl::InitNS3();
        PadDistances();
        BuildTopology();
    }

    ~NS3WiFiMultiUEChannel() { Simulator::Destroy(); }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    void transmit(int ue_id, int step_id, int packet_size = -1)
    {
        if (ue_id < 0 || ue_id >= n_ues_)
            throw std::runtime_error("ue_id out of range");
        if (packet_size < 0) packet_size = packet_size_bytes_;

        const auto uid = static_cast<uint32_t>(ue_id);
        const auto sid = static_cast<uint32_t>(step_id);

        double send_abs_ms = sim_start_ms_; //+ sid * step_duration_ms_;
        double delay_ms    = send_abs_ms - Simulator::Now().GetMilliSeconds();

        if (delay_ms > 0.0)
            Simulator::Schedule(MilliSeconds(delay_ms),
                                &NS3WiFiMultiUEChannel::DoSend, this,
                                uid, sid, packet_size);
        else
            Simulator::Schedule(NanoSeconds(1),
                                &NS3WiFiMultiUEChannel::DoSend, this,
                                uid, sid, packet_size);
    }

    std::vector<std::pair<int, int>> flush(int step_id)
    {
        double end_abs_ms = sim_start_ms_ + (step_id + 1.0) * step_duration_ms_;
        double delay_ms   = end_abs_ms - Simulator::Now().GetMilliSeconds();

        arrived_pairs_.clear();

        if (delay_ms > 0.0) {
            Simulator::Stop(MilliSeconds(delay_ms));
            Simulator::Run();
        }

        std::vector<std::pair<int, int>> out;
        out.reserve(arrived_pairs_.size());
        for (const auto& p : arrived_pairs_)
            out.push_back({static_cast<int>(p.first), static_cast<int>(p.second)});
        return out;
    }

    void reset()
    {
        Simulator::Destroy();
        nodes_ = NodeContainer();
        send_sockets_.clear();
        recv_socket_ = nullptr;
        arrived_pairs_.clear();
        PadDistances();
        BuildTopology();
    }

    py::dict get_channel_info() const
    {
        py::dict d;
        d["state"]             = "NS3_WIFI_MULTI_UE";
        d["n_ues"]             = n_ues_;
        d["step_duration_ms"]  = step_duration_ms_;
        d["tx_power_dbm"]      = tx_power_dbm_;
        d["loss_exponent"]     = loss_exponent_;
        d["max_retries"]       = max_retries_;
        d["packet_size_bytes"] = packet_size_bytes_;
        d["pending_arrivals"]  = static_cast<int>(arrived_pairs_.size());
        return d;
    }

  private:
    // -----------------------------------------------------------------------
    // Topology setup
    // -----------------------------------------------------------------------

    void PadDistances()
    {
        if (distances_m_.empty()) distances_m_.push_back(10.0);
        while (static_cast<int>(distances_m_.size()) < n_ues_)
            distances_m_.push_back(distances_m_.back());
    }

    void BuildTopology()
    {
        Ipv4AddressGenerator::Reset();

        // Nodes: index 0 = AP, indices 1..n_ues_ = STAs
        nodes_.Create(static_cast<uint32_t>(n_ues_ + 1));

        // PHY + WiFi (AP/STA infrastructure mode)
        auto cfg = netrl::ConfigureYans80211a(tx_power_dbm_, loss_exponent_);

        Ssid ssid = Ssid("netrl-bss");

        WifiMacHelper mac_ap;
        mac_ap.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
        NetDeviceContainer ap_device = cfg.wifi.Install(cfg.phy, mac_ap, nodes_.Get(0));

        NodeContainer sta_nodes;
        for (int i = 0; i < n_ues_; ++i)
            sta_nodes.Add(nodes_.Get(static_cast<uint32_t>(i + 1)));

        WifiMacHelper mac_sta;
        mac_sta.SetType("ns3::StaWifiMac",
                        "Ssid",          SsidValue(ssid),
                        "ActiveProbing", BooleanValue(true));
        NetDeviceContainer sta_devices = cfg.wifi.Install(cfg.phy, mac_sta, sta_nodes);

        Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/FrameRetryLimit",
                    UintegerValue(static_cast<uint32_t>(max_retries_)));

        // Positions: AP at origin, UE i at (distances_m_[i], 0, 0)
        std::vector<Vector> positions;
        positions.reserve(static_cast<std::size_t>(n_ues_ + 1));
        positions.emplace_back(0.0, 0.0, 0.0);
        for (int i = 0; i < n_ues_; ++i)
            positions.emplace_back(distances_m_[static_cast<std::size_t>(i)], 0.0, 0.0);
        netrl::InstallConstantPositions(nodes_, positions);

        NetDeviceContainer all_devices;
        all_devices.Add(ap_device);
        all_devices.Add(sta_devices);
        auto iface = netrl::InstallInternetStack(nodes_, all_devices);

        // Receive socket on the AP (node 0)
        recv_socket_ = netrl::CreateUdpRecvSocket(
            nodes_.Get(0), 9,
            MakeCallback(&NS3WiFiMultiUEChannel::ReceivePacket, this));

        // One send socket per UE (nodes 1..n_ues_)
        Ipv4Address ap_addr = iface.GetAddress(0);
        send_sockets_.resize(static_cast<std::size_t>(n_ues_));
        for (int i = 0; i < n_ues_; ++i)
            send_sockets_[static_cast<std::size_t>(i)] = netrl::CreateUdpSendSocket(
                nodes_.Get(static_cast<uint32_t>(i + 1)), ap_addr, 9);

        sim_start_ms_ = netrl::RunWarmup(step_duration_ms_, 500.0);
    }

    // -----------------------------------------------------------------------
    // Callbacks
    // -----------------------------------------------------------------------

    void ReceivePacket(Ptr<Socket> socket)
    {
        Ptr<Packet> pkt;
        Address from;
        while ((pkt = socket->RecvFrom(from)) != nullptr) {
            if (pkt->GetSize() >= 8) {
                uint8_t buf[8];
                pkt->CopyData(buf, 8);
                // First 4 bytes = ue_id, next 4 bytes = step_id
                uint32_t ue_id   = netrl::DecodeStepId(buf);
                uint32_t step_id = netrl::DecodeStepId(buf + 4);
                arrived_pairs_.push_back({ue_id, step_id});
            }
        }
    }

    void DoSend(uint32_t ue_id, uint32_t step_id, int pkt_size)
    {
        // Build 8-byte header: [ue_id (4B)][step_id (4B)]
        auto ue_hdr   = netrl::EncodeStepId(ue_id);
        auto step_hdr = netrl::EncodeStepId(step_id);

        std::vector<uint8_t> payload(
            static_cast<std::size_t>(std::max(pkt_size, 8)), 0);
        std::copy(ue_hdr.begin(),   ue_hdr.end(),   payload.begin());
        std::copy(step_hdr.begin(), step_hdr.end(), payload.begin() + 4);

        Ptr<Packet> pkt = Create<Packet>(payload.data(),
                                         static_cast<uint32_t>(payload.size()));
        if (ue_id < send_sockets_.size())
            send_sockets_[ue_id]->Send(pkt);
    }

    // -----------------------------------------------------------------------
    // Member variables (public for pybind11 property access)
    // -----------------------------------------------------------------------

  public:
    int                  n_ues_;
    std::vector<double>  distances_m_;
    double               step_duration_ms_;
    double               tx_power_dbm_;
    double               loss_exponent_;
    int                  max_retries_;
    int                  packet_size_bytes_;

    double                    sim_start_ms_ = 0.0;
    NodeContainer             nodes_;
    std::vector<Ptr<Socket>>  send_sockets_;
    Ptr<Socket>               recv_socket_;

    std::vector<std::pair<uint32_t, uint32_t>> arrived_pairs_;
};

// ============================================================================
// pybind11 bindings
// ============================================================================

PYBIND11_MODULE(_netrl_multi_ue_ext, m)
{
    m.doc() = "NetRL NS3 multi-UE WiFi channel (pybind11 backend)";

    py::class_<NS3WiFiMultiUEChannel>(m, "NS3WiFiMultiUEChannel")
        .def(py::init<int, const std::vector<double>&, double, double, double, int, int>(),
             py::arg("n_ues"),
             py::arg("distances_m"),
             py::arg("step_duration_ms")  = 1.0,
             py::arg("tx_power_dbm")      = 20.0,
             py::arg("loss_exponent")     = 3.0,
             py::arg("max_retries")       = 7,
             py::arg("packet_size_bytes") = 64)
        .def("transmit",
             &NS3WiFiMultiUEChannel::transmit,
             py::arg("ue_id"),
             py::arg("step_id"),
             py::arg("packet_size") = -1)
        .def("flush",            &NS3WiFiMultiUEChannel::flush, py::arg("step_id"))
        .def("reset",            &NS3WiFiMultiUEChannel::reset)
        .def("get_channel_info", &NS3WiFiMultiUEChannel::get_channel_info);
}
