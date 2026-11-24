#!/usr/bin/env python3
"""
Unit tests for LoRa-to-Local-LLM Agent Pipeline.
"""

import unittest
from lora_simulator import calculate_lora_toa_ms, generate_sensor_packet
from energy_profiler import parse_nvidia_smi_power


class TestLoraPipeline(unittest.TestCase):
    def test_lora_toa_sf7(self):
        # Standard SX1262 24-byte payload at SF7, BW=125kHz, CR=4/5 (1)
        toa = calculate_lora_toa_ms(payload_bytes=24, sf=7, bw_khz=125, cr=1)
        self.assertGreater(toa, 40.0)
        self.assertLess(toa, 80.0)

    def test_lora_toa_sf12_higher_than_sf7(self):
        toa_sf7 = calculate_lora_toa_ms(payload_bytes=24, sf=7, bw_khz=125, cr=1)
        toa_sf12 = calculate_lora_toa_ms(payload_bytes=24, sf=12, bw_khz=125, cr=1)
        self.assertGreater(toa_sf12, toa_sf7 * 10)

    def test_generate_sensor_packet_structure(self):
        pkt = generate_sensor_packet(node_id="node-01", sf=7, anomaly=True)
        self.assertEqual(pkt["node_id"], "node-01")
        self.assertEqual(pkt["sf"], 7)
        self.assertTrue(pkt["anomaly"])
        self.assertIn("toa_ms", pkt)
        self.assertIn("telemetry_text", pkt)
        self.assertGreater(pkt["vibration_rms"], 2.0)

    def test_parse_nvidia_smi_power(self):
        sample = "18.45, 52, 420"
        parsed = parse_nvidia_smi_power(sample)
        self.assertAlmostEqual(parsed["power_watts"], 18.45)
        self.assertEqual(parsed["temp_c"], 52)
        self.assertEqual(parsed["vram_used_mb"], 420)


if __name__ == "__main__":
    unittest.main()
