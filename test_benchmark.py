#!/usr/bin/env python3
"""
Unit tests for benchmark parser and markdown formatter.
"""

import unittest
from benchmark import parse_ollama_timing, format_markdown_report, detect_system_profile


class TestBenchmark(unittest.TestCase):
    def test_parse_ollama_timing(self):
        sample_data = {
            "prompt_eval_count": 50,
            "prompt_eval_duration": 1_000_000_000,  # 1 second in nanoseconds
            "eval_count": 100,
            "eval_duration": 2_000_000_000,         # 2 seconds in nanoseconds
            "load_duration": 500_000_000,           # 0.5 seconds
        }
        metrics = parse_ollama_timing(sample_data)
        self.assertEqual(metrics["prompt_tokens"], 50)
        self.assertEqual(metrics["prompt_speed_tok_s"], 50.0)
        self.assertEqual(metrics["eval_tokens"], 100)
        self.assertEqual(metrics["eval_speed_tok_s"], 50.0)
        self.assertEqual(metrics["load_time_s"], 0.5)

    def test_detect_system_profile(self):
        profile = detect_system_profile()
        self.assertIn("cpu", profile)
        self.assertIn("ram", profile)
        self.assertIn("gpu_model", profile)
        self.assertIn("os", profile)

    def test_format_markdown_report(self):
        profile = {
            "cpu": "Intel Core i5-11300H",
            "ram": "22.0 GiB",
            "gpu_model": "NVIDIA GeForce GTX 1650",
            "gpu_vram": "4.0 GiB",
            "gpu_driver": "595.99.02",
            "cuda_version": "13.2",
            "os": "Linux-7.0.0-43-generic",
            "timestamp": "2026-11-24T18:00:00Z",
        }
        results = [
            {
                "model": "llama3.2:1b",
                "runs": 3,
                "avg_prompt_speed_tok_s": 85.5,
                "avg_eval_speed_tok_s": 42.1,
            },
            {
                "model": "llama3.1:8b",
                "runs": 3,
                "avg_prompt_speed_tok_s": 18.2,
                "avg_eval_speed_tok_s": 5.4,
            },
        ]
        md = format_markdown_report(profile, results)
        self.assertIn("Intel Core i5-11300H", md)
        self.assertIn("GTX 1650", md)
        self.assertIn("llama3.2:1b", md)
        self.assertIn("llama3.1:8b", md)
        self.assertIn("Full GPU VRAM Offload", md)
        self.assertIn("Hybrid Offload", md)


if __name__ == "__main__":
    unittest.main()
