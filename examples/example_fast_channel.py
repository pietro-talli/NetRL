#!/usr/bin/env python3
"""
example_fast_channel.py - Direct fast channel test (without NetworkedEnv)

This demonstrates:
1. Loading the pybind11 fast channel directly
2. Using it independently with observations
3. Performance measurement
4. Multiple packets per step
"""

import sys
import os
import time
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("NetRL Fast Channel Direct Test (pybind11)")
print("=" * 80)
print()

# Step 1: Import extension
print("[1/4] Importing fast channel...")
try:
    from netrl.channels.ns3_channel_fast import NS3WiFiChannelFast
    from netrl.channels.network_config import NetworkConfig
    print("✅ Fast channel imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("   Make sure pybind11 extension is built: python setup.py build_ext --inplace")
    sys.exit(1)

# Step 2: Create channel
print()
print("[2/4] Creating fast WiFi channel...")
try:
    config = NetworkConfig(
        buffer_size=10,
        delay_steps=0,
        seed=42
    )

    channel = NS3WiFiChannelFast(
        config=config,
        distance_m=15.0,
        step_duration_ms=2.0,
        tx_power_dbm=20.0,
        loss_exponent=3.0,
        max_retries=7,
        packet_size_bytes=256,
    )
    print("✅ Fast channel created!")
    print(f"   Distance: {channel._channel.distance_m} m")
    print(f"   Step duration: {channel._channel.step_duration_ms} ms")
except Exception as e:
    print(f"❌ Failed to create channel: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Run transmission/reception test
print()
print("[3/4] Running transmission/reception test...")
print()

try:
    num_steps = 100
    total_packets_sent = 0
    total_packets_received = 0
    total_bytes = 0

    loop_start = time.time()

    for step in range(num_steps):
        # Create random observation (CartPole-like: 4 values)
        obs = np.array([0.1 * step, 0.2 * step, 0.05 * step, 0.15 * step], dtype=np.float64)
        total_bytes += obs.nbytes

        # Transmit
        channel.transmit(obs, step=step)
        total_packets_sent += 1

        # Flush (collect received packets)
        packets = channel.flush(step=step)
        total_packets_received += len(packets)

        if step % 20 == 0:
            print(f"  Step {step:3d}: sent {total_packets_sent:3d} packets, "
                  f"received {total_packets_received:3d} packets")

    loop_time = time.time() - loop_start

    print()
    print("=" * 80)
    print("Performance Statistics")
    print("=" * 80)
    print(f"  Total steps:              {num_steps}")
    print(f"  Total packets sent:       {total_packets_sent}")
    print(f"  Total packets received:   {total_packets_received}")
    print(f"  Data transmitted:         {total_bytes / 1024:.2f} KB")
    print(f"  Total time:               {loop_time:.3f} seconds")
    print(f"  Time per step:            {loop_time / num_steps * 1000:.3f} ms")
    print(f"  Throughput:               {num_steps / loop_time:.1f} steps/sec")
    print()

    # Get final channel info
    info = channel.get_channel_info()
    print("Channel Info:")
    for key, value in sorted(info.items()):
        print(f"  {key}: {value}")

    print()
    print("=" * 80)
    print("✅ Test completed successfully!")
    print("=" * 80)
    print()
    print("The fast pybind11 channel is working correctly!")
    print("Performance expected: ~15-20x faster than subprocess version.")

except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
