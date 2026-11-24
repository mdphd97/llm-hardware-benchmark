#!/usr/bin/env python3
"""
Unit tests for multi-system benchmark parser, slug generation, CPU/GPU layer modes, and aggregator.
"""

import json
import os
import shutil
import tempfile
import unittest
from benchmark import (
    aggregate_all_systems,
    detect_system_profile,
    determine_execution_mode,
    parse_ollama_timing,
    sanitize_slug,
)


class TestMultiSystemBenchmark(unittest.TestCase):
    def test_sanitize_slug(self):
        self.assertEqual(sanitize_slug("GTX 1650 (4GB)"), "gtx_1650_4gb")
        self.assertEqual(sanitize_slug("Apple M2 Pro @ 16GB"), "apple_m2_pro_16gb")
        self.assertEqual(sanitize_slug("Intel i5-11300H"), "intel_i5_11300h")

    def test_determine_execution_mode(self):
        self.assertEqual(
            determine_execution_mode("llama3.2:1b", 0),
            "CPU + System RAM (0 GPU layers)",
        )
        self.assertEqual(
            determine_execution_mode("llama3.1:8b", 16),
            "Partial Offload (16 GPU layers)",
        )
        self.assertEqual(
            determine_execution_mode("llama3.2:3b", None),
            "Full VRAM Offload",
        )
        self.assertEqual(
            determine_execution_mode("llama3.1:8b", None),
            "Hybrid Offload (VRAM + System RAM)",
        )

    def test_parse_ollama_timing(self):
        sample_data = {
            "prompt_eval_count": 50,
            "prompt_eval_duration": 1_000_000_000,
            "eval_count": 100,
            "eval_duration": 2_000_000_000,
            "load_duration": 500_000_000,
        }
        metrics = parse_ollama_timing(sample_data)
        self.assertEqual(metrics["prompt_tokens"], 50)
        self.assertEqual(metrics["prompt_speed_tok_s"], 50.0)
        self.assertEqual(metrics["eval_tokens"], 100)
        self.assertEqual(metrics["eval_speed_tok_s"], 50.0)
        self.assertEqual(metrics["load_time_s"], 0.5)

    def test_detect_system_profile_cpu_ram_mode(self):
        profile = detect_system_profile(num_gpu=0)
        self.assertTrue(profile["system_id"].startswith("cpu_ram_"))
        self.assertEqual(profile["num_gpu_layers"], 0)

    def test_aggregate_all_systems(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            sys_a = os.path.join(tmp_dir, "sys_gtx1650")
            os.makedirs(sys_a, exist_ok=True)
            payload_a = {
                "system_profile": {
                    "system_id": "sys_gtx1650",
                    "num_gpu_layers": 0,
                },
                "benchmark_results": [
                    {
                        "model": "llama3.2:1b",
                        "execution_mode": "CPU + System RAM (0 GPU layers)",
                        "avg_prompt_speed_tok_s": 85.0,
                        "avg_eval_speed_tok_s": 22.5,
                    }
                ],
            }
            with open(os.path.join(sys_a, "benchmark_results.json"), "w") as f:
                json.dump(payload_a, f)

            leaderboard = aggregate_all_systems(results_root=tmp_dir)
            self.assertIn("Cross-System Hardware Leaderboard", leaderboard)
            self.assertIn("sys_gtx1650", leaderboard)
            self.assertIn("CPU + System RAM (0 GPU layers)", leaderboard)
            self.assertIn("22.5", leaderboard)
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
