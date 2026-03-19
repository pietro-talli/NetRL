"""
NS3WiFiChannelFastConfig - Configuration for pybind11-based fast WiFi channel.

This config is used with NetworkedEnv to select the fast pybind11 binding
instead of the subprocess-based NS3WifiChannel.

Example Usage
=============
    from netrl import NetworkedEnv, NetworkConfig
    from netrl.channels.ns3_wifi_fast_config import NS3WiFiChannelFastConfig

    config = NetworkConfig(buffer_size=10, seed=42)
    wifi_fast_config = NS3WiFiChannelFastConfig(distance_m=15.0, step_duration_ms=2.0)

    env = NetworkedEnv(
        gym.make("CartPole-v1"),
        config,
        channel_config=wifi_fast_config
    )

This is 15-20x faster than the subprocess version!
"""

from dataclasses import dataclass


@dataclass
class NS3WiFiChannelFastConfig:
    """
    Configuration for the fast NS3WiFiChannel (pybind11 binding).

    This is a direct C++ binding with no subprocess overhead.

    Parameters
    ----------
    distance_m : float
        STA-to-AP distance in metres. Default: 15.0

    step_duration_ms : float
        Environment step duration in milliseconds. Default: 2.0

    tx_power_dbm : float
        TX power in dBm. Default: 20.0

    loss_exponent : float
        Path-loss exponent. Default: 3.0

    max_retries : int
        Maximum MAC retransmission attempts. Default: 7

    packet_size_bytes : int
        Default packet size in bytes. Default: 256
    """

    distance_m: float = 15.0
    step_duration_ms: float = 2.0
    tx_power_dbm: float = 20.0
    loss_exponent: float = 3.0
    max_retries: int = 7
    packet_size_bytes: int = 256
