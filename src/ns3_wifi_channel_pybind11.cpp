/**
 * ns3_wifi_channel_pybind11.cpp
 *
 * Single-UE WiFi channel simulator using pybind11 for Python integration.
 * Replaces the subprocess-based ns3_wifi_sim with direct in-process C++ calls.
 *
 * Shared NS3 boilerplate (YANS config, mobility, internet stack, sockets,
 * packet encoding, warmup) lives in ns3_channel_utils.h.
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
#include <deque>
#include <vector>

namespace py = pybind11;
using namespace ns3;

// ============================================================================
// WiFi Channel Implementation
// ============================================================================

class NS3WiFiChannel {
  private:

  public:
    NS3WiFiChannel(double   distance_m        = 15.0,
                   double   step_duration_ms  = 2.0,
                   double   tx_power_dbm      = 20.0,
                   double   loss_exponent     = 3.0,
                   int      max_retries       = 7,
                   int      packet_size_bytes = 256,
                   uint64_t seed              = 0)
        : distance_m_(distance_m),
          step_duration_ms_(step_duration_ms),
          tx_power_dbm_(tx_power_dbm),
          loss_exponent_(loss_exponent),
          max_retries_(max_retries),
          packet_size_bytes_(packet_size_bytes),
          seed_(seed)
    {
        netrl::InitNS3();
        BuildTopology();
    }

    ~NS3WiFiChannel() { Simulator::Destroy(); }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    void transmit(int step, int packet_size = -1)
    {
        if (packet_size < 0) packet_size = packet_size_bytes_;

        double send_time_ms = sim_start_ms_ + step * step_duration_ms_
                              + step_duration_ms_ * 0.01;
        double delay_ms = send_time_ms - Simulator::Now().GetMilliSeconds();

        if (delay_ms > 0.0)
            Simulator::Schedule(MilliSeconds(delay_ms),
                                &NS3WiFiChannel::DoSend, this, step, packet_size);
        else
            Simulator::Schedule(NanoSeconds(1),
                                &NS3WiFiChannel::DoSend, this, step, packet_size);
    }

    std::vector<int> flush(int step)
    {
        double flush_time_ms = sim_start_ms_ + (step + 1.0) * step_duration_ms_;
        double delay_ms = flush_time_ms - Simulator::Now().GetMilliSeconds();

        if (delay_ms > 0.0) {
            Simulator::Stop(MilliSeconds(delay_ms));
            Simulator::Run();
        }

        std::vector<int> result(arrived_ids_.begin(), arrived_ids_.end());
        arrived_ids_.clear();
        return result;
    }

    void reset()
    {
        Simulator::Destroy();
        send_socket_ = nullptr;
        recv_socket_ = nullptr;
        nodes_       = NodeContainer();
        arrived_ids_.clear();
        BuildTopology();
    }

    py::dict get_channel_info() const
    {
        py::dict d;
        d["state"]            = "NS3_WIFI";
        d["distance_m"]       = distance_m_;
        d["step_duration_ms"] = step_duration_ms_;
        d["tx_power_dbm"]     = tx_power_dbm_;
        d["loss_exponent"]    = loss_exponent_;
        d["max_retries"]      = max_retries_;
        d["packet_size_bytes"]= packet_size_bytes_;
        d["pending_tx_count"] = static_cast<int>(arrived_ids_.size());
        return d;
    }

  private:
    // -----------------------------------------------------------------------
    // Topology setup
    // -----------------------------------------------------------------------

    void BuildTopology()
    {
        Ipv4AddressGenerator::Reset();
        nodes_.Create(2);

        // PHY + WiFi (ad-hoc MAC for single-STA simplicity)
        auto cfg = netrl::ConfigureYans80211a(tx_power_dbm_, loss_exponent_, 40.0);
        WifiMacHelper mac;
        mac.SetType("ns3::AdhocWifiMac");
        NetDeviceContainer devices = cfg.wifi.Install(cfg.phy, mac, nodes_);

        Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/FrameRetryLimit",
                    UintegerValue(static_cast<uint32_t>(max_retries_)));

        netrl::InstallConstantPositions(
            nodes_, {Vector(0.0, 0.0, 0.0), Vector(distance_m_, 0.0, 0.0)});

        auto iface = netrl::InstallInternetStack(nodes_, devices);

        recv_socket_ = netrl::CreateUdpRecvSocket(
            nodes_.Get(1), 9,
            MakeCallback(&NS3WiFiChannel::ReceivePacket, this));
        send_socket_ = netrl::CreateUdpSendSocket(
            nodes_.Get(0), iface.GetAddress(1), 9);

        sim_start_ms_ = netrl::RunWarmup(step_duration_ms_, 310.0);
    }

    // -----------------------------------------------------------------------
    // Callbacks
    // -----------------------------------------------------------------------

    void ReceivePacket(Ptr<Socket> /*socket*/)
    {
        Ptr<Packet> pkt;
        Address from;
        while ((pkt = recv_socket_->RecvFrom(from)) != nullptr) {
            if (pkt->GetSize() >= 4) {
                uint8_t buf[4];
                pkt->CopyData(buf, 4);
                arrived_ids_.push_back(netrl::DecodeStepId(buf));
            }
        }
    }

    void DoSend(uint32_t step_id, int packet_size)
    {
        send_socket_->Send(netrl::CreateProbePacket(step_id, packet_size));
    }

    // -----------------------------------------------------------------------
    // Member variables (public for pybind11 property access)
    // -----------------------------------------------------------------------

  public:
    double   distance_m_;
    double   step_duration_ms_;
    double   tx_power_dbm_;
    double   loss_exponent_;
    int      max_retries_;
    int      packet_size_bytes_;
    uint64_t seed_;

    double        sim_start_ms_ = 0.0;
    NodeContainer nodes_;
    Ptr<Socket>   send_socket_;
    Ptr<Socket>   recv_socket_;

    std::deque<uint32_t> arrived_ids_;
};

// ============================================================================
// pybind11 bindings
// ============================================================================

PYBIND11_MODULE(_netrl_ext, m)
{
    m.doc() = "NetRL NS3 single-UE WiFi channel (pybind11 backend)";

    py::class_<NS3WiFiChannel>(m, "NS3WiFiChannel")
        .def(py::init<double, double, double, double, int, int, uint64_t>(),
             py::arg("distance_m")        = 15.0,
             py::arg("step_duration_ms")  = 2.0,
             py::arg("tx_power_dbm")      = 20.0,
             py::arg("loss_exponent")     = 3.0,
             py::arg("max_retries")       = 7,
             py::arg("packet_size_bytes") = 256,
             py::arg("seed")              = 0)
        .def("transmit",
             &NS3WiFiChannel::transmit,
             py::arg("step"),
             py::arg("packet_size") = -1)
        .def("flush",  &NS3WiFiChannel::flush,  py::arg("step"))
        .def("reset",  &NS3WiFiChannel::reset)
        .def("get_channel_info", &NS3WiFiChannel::get_channel_info)
        .def_property_readonly("distance_m",
            [](const NS3WiFiChannel& c) { return c.distance_m_; })
        .def_property_readonly("step_duration_ms",
            [](const NS3WiFiChannel& c) { return c.step_duration_ms_; });
}
