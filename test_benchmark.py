#!/usr/bin/env python3
"""
Unit tests for multi-system benchmark parser, slug generation, and aggregator.
"""

import json
import os
import shutil
import tempfile
import unittest
from benchmark import (
    aggregate_all_systems,
    detect_system_profile,
    format_system_markdown_table,
    parse_ollama_timing,
    sanitize_slug,
)


class TestMultiSystemBenchmark(unittest.TestCase):
    def test_sanitize_slug(self):
        self.assertEqual(sanitize_slug("GTX 1650 (4GB)"), "gtx_1650_4gb")
        self.assertEqual(sanitize_slug("Apple M2 Pro @ 16GB"), "apple_m2_pro_16gb")
        self.assertEqual(sanitize_slug("Intel i5-11300H"), "intel_i5_11300h")

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

    def test_detect_system_profile_custom_name(self):
        profile = detect_system_profile(custom_name="Office Workstation RTX 4090")
        self.assertEqual(profile["system_id"], "office_workstation_rtx_4090")

    def test_aggregate_all_systems(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            sys_a = os.path.join(tmp_dir, "sys_gtx1650")
            os.makedirs(sys_a, exist_ok=True)
            payload_a = {
                "system_profile": {
                    "system_id": "sys_gtx1650",
                    "cpu": "Intel i5",
                    "ram": "24 GiB",
                    "gpu_model": "GTX 1650",
                    "gpu_vram": "4 GiB",
                },
                "benchmark_results": [
                    {
                        "model": "llama3.2:1b",
                        "avg_prompt_speed_tok_s": 3800.0,
                        "avg_eval_speed_tok_s": 105.0,
                    }
                ],
            }
            with open(os.path.join(sys_a, "benchmark_results.json"), "w") as f:
                json.dump(payload_a, f)

            leaderboard = aggregate_all_systems(results_root=tmp_dir)
            self.assertIn("Cross-System Hardware Leaderboard", leaderboard)
            self.assertIn("sys_gtx1650", leaderboard)
            self.assertIn("llama3.2:1b", leaderboard)
            self.assertIn("105.0", leaderboard)
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
