<div align="center">

![LLM Hardware Performance Benchmark](assets/hero_banner.svg)

<br/>

[![Python 3](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](benchmark.py)
[![Ollama](https://img.shields.io/badge/Ollama-0.40%2B-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](results/gtx_1650_i5_11300h/summary.md)
[![Vulkan](https://img.shields.io/badge/Vulkan-Intel%20Iris%20Xe-AC162C?style=for-the-badge&logo=vulkan&logoColor=white)](results/intel_iris_xe_vulkan/summary.md)

**A reproducible, multi-system benchmark suite evaluating local LLM throughput across discrete GPUs, CPU system memory, and integrated GPUs.**

[Leaderboard](#-cross-system--memory-tier-leaderboard) · [Memory Topology](#-memory-topology--architecture) · [How to Replicate](#-how-to-replicate-on-any-machine) · [Full Blog Post](BLOG_POST.md)

</div>

---

## 📊 Visual Performance Overview

![Token Generation Throughput Across Compute Tiers](assets/compute_tier_comparison.svg)

---

## 🏆 Cross-System & Memory-Tier Leaderboard

All benchmarks run with **zero-temperature greedy decoding** across a fixed **128-token output ceiling** after runner eviction and warm-up caching:

| System ID | Compute / Memory Tier | Model | Prompt (tok/s) | Generation (tok/s) |
| :--- | :--- | :--- | :---: | :---: |
| `gtx_1650_i5_11300h` | **Full VRAM Offload** (CUDA GTX 1650 4 GB) | `llama3.2:1b` | `3,121.19` | **105.03** |
| `cpu_ram_i5_11300h_310ghz` | **CPU + System RAM** (`--num-gpu 0`) | `llama3.2:1b` | `701.53` | **21.80** |
| `intel_iris_xe_vulkan` | **Intel Iris Xe iGPU** (Vulkan Unified RAM) | `llama3.2:1b` | `302.92` | **6.96** |
| `gtx_1650_i5_11300h` | **Full VRAM Offload** (CUDA GTX 1650 4 GB) | `llama3.2:3b` | `1,763.39` | **63.13** |
| `cpu_ram_i5_11300h_310ghz` | **CPU + System RAM** (`--num-gpu 0`) | `llama3.2:3b` | `488.92` | **17.02** |
| `intel_iris_xe_vulkan` | **Intel Iris Xe iGPU** (Vulkan Unified RAM) | `llama3.2:3b` | `199.48` | **5.14** |
| `gtx_1650_i5_11300h` | **Hybrid Offload** (VRAM + System RAM) | `llama3.1:8b` | `169.44` | **8.18** |
| `cpu_ram_i5_11300h_310ghz` | **CPU + System RAM** (`--num-gpu 0`) | `llama3.1:8b` | `136.69` | **6.29** |
| `intel_iris_xe_vulkan` | **Intel Iris Xe iGPU** (Vulkan Unified RAM) | `llama3.1:8b` | `78.28` | **3.34** |

> [!TIP]
> **Why CPU Outperforms the Integrated GPU (~3× Faster):** On Intel Tiger Lake (`i5-11300H`), CPU cores leverage **AVX-512** and **VNNI** vector extensions paired with GGML's `CPU_REPACK` kernels. Because both the CPU and the Intel Iris Xe iGPU share the same dual-channel DDR4 memory bus, CPU inference (`21.80 tok/s`) outperforms generic Vulkan shaders on the iGPU (`6.96 tok/s`)—explaining why Ollama drops integrated GPUs by default unless `OLLAMA_IGPU_ENABLE=1` is set.

---

## 🧠 Memory Topology & Architecture

```mermaid
flowchart LR
    subgraph Discrete["Discrete GPU Tier (Fastest)"]
        VRAM["NVIDIA GTX 1650<br/>4 GB GDDR6 (~192 GB/s)"]
    end

    subgraph Bus["Interconnect"]
        PCIe["PCIe Gen3/4 Link<br/>(Hybrid Spillover Bottleneck)"]
    end

    subgraph Shared["Shared System Memory Tier (22.8 GiB DDR4)"]
        CPU["Intel i5-11300H CPU<br/>AVX-512 + VNNI (~21.8 tok/s)"]
        iGPU["Intel Iris Xe iGPU<br/>Vulkan 15.4 GiB (~6.96 tok/s)"]
    end

    VRAM <-->|"8B Hybrid Split"| PCIe
    PCIe <--> Shared
```

---

## 🚀 How to Replicate on Any Machine

Clone this repository on any Linux, macOS, or Windows (WSL) workstation running Ollama. The `--delete-after` flag automatically pulls missing test models, executes the benchmark, updates the leaderboard, and deletes the weights from disk afterward.

```bash
git clone https://github.com/mdphd97/llm-hardware-benchmark.git
cd llm-hardware-benchmark

# 1. Benchmark default compute (CUDA / Metal / Auto)
python3 benchmark.py --models llama3.2:1b llama3.2:3b llama3.1:8b --delete-after

# 2. Force pure CPU + System RAM execution (0 GPU layers)
python3 benchmark.py --num-gpu 0 --models llama3.2:1b llama3.2:3b llama3.1:8b --delete-after

# 3. Regenerate SVG visual charts from updated results/
python3 generate_charts.py
```

<details>
<summary><b>▶ Repository Structure, Rig Specs & Unit Tests</b></summary>

### Test Rig Specifications (`mdphd`)
- **CPU:** 11th Gen Intel Core i5-11300H @ 3.10GHz (4 Cores / 8 Threads)
- **RAM:** 22.8 GiB DDR4
- **Discrete GPU:** NVIDIA GeForce GTX 1650 Mobile (4,096 MiB GDDR6, CUDA 13.2, Driver 595.99.02)
- **Integrated GPU:** Intel TigerLake-LP GT2 [Iris Xe Graphics] (15.4 GiB Unified via Vulkan)
- **OS:** Ubuntu Linux (`x86_64`) | **Runtime:** Ollama `0.40.12`

### Directory Layout
```text
├── assets/                     # Auto-generated SVG banners & charts
├── results/                    # Per-system JSON logs & cross-hardware LEADERBOARD.md
├── benchmark.py                # Zero-dependency multi-system benchmark & runner eviction tool
├── generate_charts.py          # Zero-dependency SVG chart generator
├── test_benchmark.py           # Unit test suite
├── BLOG_POST.md                # Publication-ready technical review article
└── README.md
```

### Run Unit Tests
```bash
python3 -m unittest test_benchmark.py
```

</details>
