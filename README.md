# Multi-System LLM Hardware Benchmark Suite

A standardized, multi-system benchmarking framework and leaderboard evaluating local Large Language Model (LLM) inference performance across consumer hardware, GPUs, and memory architectures.

---

## Cross-System & Memory-Tier Leaderboard

All benchmarks are run using zero-temperature greedy decoding across a fixed 128-token generation window after a warm-up cycle and runner eviction:

| System ID | Compute / Memory Tier | Model | Prompt (tok/s) | Generation (tok/s) |
| :--- | :--- | :--- | :---: | :---: |
| `gtx_1650_i5_11300h` | Full VRAM Offload (CUDA GTX 1650) | `llama3.2:1b` | **3,121.19** | **105.03** |
| `cpu_ram_i5_11300h_310ghz` | CPU + System RAM (`--num-gpu 0`) | `llama3.2:1b` | **701.53** | **21.80** |
| `intel_iris_xe_vulkan` | Intel Iris Xe iGPU (Vulkan Unified RAM) | `llama3.2:1b` | **302.92** | **6.96** |
| `gtx_1650_i5_11300h` | Full VRAM Offload (CUDA GTX 1650) | `llama3.2:3b` | **1,763.39** | **63.13** |
| `cpu_ram_i5_11300h_310ghz` | CPU + System RAM (`--num-gpu 0`) | `llama3.2:3b` | **488.92** | **17.02** |
| `intel_iris_xe_vulkan` | Intel Iris Xe iGPU (Vulkan Unified RAM) | `llama3.2:3b` | **199.48** | **5.14** |
| `gtx_1650_i5_11300h` | Hybrid Offload (VRAM + System RAM) | `llama3.1:8b` | **169.44** | **8.18** |
| `cpu_ram_i5_11300h_310ghz` | CPU + System RAM (`--num-gpu 0`) | `llama3.1:8b` | **136.69** | **6.29** |
| `intel_iris_xe_vulkan` | Intel Iris Xe iGPU (Vulkan Unified RAM) | `llama3.1:8b` | **78.28** | **3.34** |

> [!TIP]
> **Why CPU Beats the Integrated GPU:** On Intel Tiger Lake (i5-11300H), the CPU cores feature AVX-512 and VNNI vector instructions paired with GGML `CPU_REPACK` kernels. Because both the CPU and the Intel Iris Xe iGPU share the same dual-channel DDR4 memory bus, CPU execution (`21.80 tok/s`) outperforms Vulkan iGPU execution (`6.96 tok/s`) by ~3x—explaining why Ollama disables integrated GPUs by default (`OLLAMA_IGPU_ENABLE=1`).

---

## LoRa-to-Local-LLM Agent Pipeline Benchmark (`lora-agent-pipeline/`)

Measures the complete **Sensor-to-Agent Latency Waterfall** and **Host Energy Consumption**:
Simulated Semtech SX1262 LoRa Telemetry $\rightarrow$ Ollama Embedding (`nomic-embed-text`) $\rightarrow$ Dockerized Qdrant Vector Retrieval $\rightarrow$ MCP Tool-Calling Loop (`llama3.2:1b`).

### Verified Waterfall Results (GTX 1650 + Ubuntu Linux)
- **Burst Profile:** 20 LoRa nodes (SX1262, SF7, 24-byte payload), 7 bearing anomalies triaged
- **LoRa Time-on-Air:** `61.70 ms`
- **Ollama Embedding (`nomic-embed-text`):** `527.12 ms`
- **Qdrant Vector Upsert / Search:** `30.26 ms` / `1.78 ms`
- **LLM MCP Reasoning (`llama3.2:1b`):** `1,012.04 ms`
- **Total End-to-End Anomaly Triage:** **`1,602.64 ms`** (~1.60 s)
- **Host GPU Power & Energy:** `21.47 W` avg | `393.81 J` total | Peak `48 °C`

### Run the LoRa Pipeline Stress Test
```bash
cd lora-agent-pipeline
docker compose up -d
python3 agent_bridge.py --nodes 20 --sf 7 --llm-model llama3.2:1b --delete-after
docker compose down -v
```

---

## How to Test a New Machine (1-Line Replication)

You can clone this repository onto any new computer (Linux, macOS, or Windows with WSL) and benchmark it immediately:

```bash
git clone https://github.com/mdphd97/llm-hardware-benchmark.git
cd llm-hardware-benchmark

# Run benchmark, record system specs, and automatically delete test weights after testing
python3 benchmark.py --models llama3.2:1b llama3.2:3b --delete-after
```

### Optional Custom System Identifier & Memory-Tier Modes
By default, the script generates a system identifier from your GPU and CPU model. You can specify a custom name or force CPU/System-RAM execution:
```bash
# Force pure CPU + System DDR4 RAM execution (0 GPU layers)
python3 benchmark.py --num-gpu 0 --models llama3.2:1b llama3.2:3b llama3.1:8b --delete-after

# Benchmark with a custom system name and backend label
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
