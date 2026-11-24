# LLM Hardware Benchmark: Running Local AI on 4 GB VRAM

Reproducible technical review and benchmark suite measuring Large Language Model (LLM) inference performance on entry-level hardware: **Intel Core i5-11300H**, **NVIDIA GeForce GTX 1650 (4 GB VRAM)**, and **24 GB DDR4 RAM** running Ubuntu Linux.

---

## Technical Review Summary

When deploying open-weight models locally, hardware memory topology dictates usability. This benchmark evaluates three operational execution tiers on consumer hardware:

1. **Full VRAM Offload (1B - 3B Models):** Models such as Llama 3.2 (1B and 3B) and Qwen 2.5 (0.5B to 3B) fit entirely inside the GTX 1650's 4 GB frame buffer. These models deliver conversational real-time speeds (30–60 tokens per second) with instant prompt evaluation.
2. **Hybrid Layer Offload (7B - 8B Models):** Models such as Llama 3.1 8B (Q4_K_M) exceed 4 GB VRAM and require splitting layers between GPU VRAM and DDR4 system memory. Offloading approximately 14 to 18 layers to the GPU yields acceptable interactive reading speeds (~5 to 8 tokens per second), but experiences latency overhead during prompt processing due to PCIe bus data transfer.
3. **Pure System Memory (14B+ Models):** Larger models execute almost entirely within the 24 GB DDR4 system memory. At 1.5 to 2.5 tokens per second, these operate effectively as asynchronous background processing tasks rather than real-time interactive chat.

---

## Test Rig Specifications

| Component | Specification |
| :--- | :--- |
| **Processor (CPU)** | Intel Core i5-11300H @ 3.10GHz (4 Cores / 8 Threads) |
| **System Memory (RAM)** | 24 GB DDR4 |
| **Dedicated Graphics (GPU)** | NVIDIA GeForce GTX 1650 (TU117 Turing) |
| **Video Memory (VRAM)** | 4,096 MiB (4.0 GiB) GDDR6 |
| **Driver & CUDA** | NVIDIA Driver 595.99.02 \| CUDA 13.2 |
| **Operating System** | Ubuntu Linux (x86_64) |
| **Inference Engine** | Ollama 0.40.12 (llama.cpp backend) |

---

## How to Replicate This Benchmark

This repository requires only Python 3 and Ollama. Any user can replicate this benchmark on Linux, Windows, or macOS.

### 1. Pull the Benchmark Models
```bash
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull llama3.1:8b
```

### 2. Run the Benchmark Tool
```bash
python3 benchmark.py --models llama3.2:1b llama3.2:3b llama3.1:8b --runs 3
```

The script will automatically:
- Detect your system's CPU, GPU, VRAM, and driver configuration.
- Execute a warm-up cycle to ensure model weights are cached.
- Perform multiple evaluation passes using fixed prompts and zero-temperature greedy decoding.
- Save raw data to `results/benchmark_results.json` and output a formatted markdown table to `results/benchmark_table.md`.

---

## Repository Structure

```text
├── AGENTS.md               # Autonomous agent instructions and architecture rules
├── BLOG_POST.md            # Long-form technical review article ready for publication
├── benchmark.py            # Zero-dependency benchmark runner and hardware profiler
├── test_benchmark.py       # Unit test suite verifying timing calculations and report formatting
├── results/
│   ├── benchmark_results.json # Raw timing logs (prompt eval ns, eval ns, tokens)
│   └── benchmark_table.md     # Formatted GitHub markdown table
└── README.md
```

## Running Unit Tests

```bash
python3 -m unittest test_benchmark.py
```
