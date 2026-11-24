# LLM Hardware Benchmark Project Rules

## Overview
This repository provides automated, reproducible inference benchmarking scripts and technical reports evaluating local Large Language Model (LLM) performance across everyday consumer hardware configurations (NVIDIA GeForce GTX 1650 4GB VRAM + 24GB DDR4 system RAM).

## Development & Execution Guidelines
- Standard Python 3 standard library only for `benchmark.py` (zero external dependencies required for replication).
- Ollama local API serves as the primary inference runtime.
- Never hardcode dynamic metrics; always parse exact nanosecond timestamps from the runtime response.
- All benchmark test modifications must be accompanied by unit tests in `test_benchmark.py`.
