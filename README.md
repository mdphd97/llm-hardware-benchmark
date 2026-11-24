# Multi-System LLM Hardware Benchmark Suite

A standardized, multi-system benchmarking framework and leaderboard evaluating local Large Language Model (LLM) inference performance across consumer hardware, GPUs, and memory architectures.

---

## Cross-System Leaderboard

All benchmarks are run using zero-temperature greedy decoding across a fixed 128-token generation window after a warm-up cycle:

| System ID | GPU / Compute | CPU & RAM | Model | Prompt (tok/s) | Generation (tok/s) |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `gtx_1650_i5_11300h` | GTX 1650 (4.0 GiB) | Intel i5-11300H (22.8 GiB DDR4) | `llama3.2:1b` | **3,842.36** | **106.17** |
| `gtx_1650_i5_11300h` | GTX 1650 (4.0 GiB) | Intel i5-11300H (22.8 GiB DDR4) | `llama3.2:3b` | **2,251.89** | **62.99** |

---

## How to Test a New Machine (1-Line Replication)

You can clone this repository onto any new computer (Linux, macOS, or Windows with WSL) and benchmark it immediately:

```bash
git clone https://github.com/mdphd97/llm-hardware-benchmark.git
cd llm-hardware-benchmark

# Run benchmark, record system specs, and automatically delete test weights after testing
python3 benchmark.py --models llama3.2:1b llama3.2:3b --delete-after
```

### Optional Custom System Identifier
By default, the script generates a system identifier from your GPU and CPU model. You can specify a custom name:
```bash
python3 benchmark.py --system-name "work-m2-macbook-pro" --models llama3.2:1b llama3.2:3b --delete-after
```

### What Happens Automatically
1. Probes the host system (NVIDIA GPU via `nvidia-smi`, Apple Silicon Metal via `sysctl`, or CPU/RAM).
2. Checks Ollama for required models; pulls missing models automatically.
3. Performs a warm-up inference pass to ensure weights and context structures are cached.
4. Executes multiple measurement passes measuring prompt processing throughput (TTFT) and token generation speed.
5. Saves the raw system report into `results/<system-id>/benchmark_results.json` and `results/<system-id>/summary.md`.
6. Automatically cleans up test models from disk if `--delete-after` is passed.
7. Re-aggregates all systems in `results/*/` and updates the master leaderboard.

---

## Repository Structure

```text
├── AGENTS.md               # Autonomous development and architecture instructions
├── BLOG_POST.md            # Publication-ready technical review article
├── benchmark.py            # Zero-dependency multi-system benchmark runner & aggregator
├── test_benchmark.py       # Unit test suite verifying parsers, slugs, and aggregators
├── results/
│   ├── LEADERBOARD.md      # Auto-generated comparative leaderboard across all machines
│   └── gtx_1650_i5_11300h/ # Machine-specific dataset
│       ├── benchmark_results.json
│       └── summary.md
└── README.md
```

## Running the Unit Test Suite

```bash
python3 -m unittest test_benchmark.py
```
