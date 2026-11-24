### System: `gtx_1650_i5_11300h`

- **Processor (CPU):** 11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz
- **System Memory (RAM):** 22.8 GiB
- **Graphics (GPU):** NVIDIA GeForce GTX 1650 (4.0 GiB (4096 MiB))
- **GPU Layers (`num_gpu`):** auto
- **OS:** Linux-7.0.0-43-generic-x86_64-with-glibc2.43
- **Benchmark Date:** 2026-11-24T18:40:19.781123+00:00

| Model Name | Prompt Processing (tok/s) | Generation Speed (tok/s) | Runs | Execution Mode |
| :--- | :---: | :---: | :---: | :--- |
| `llama3.2:1b` | **3121.19** | **105.03** | 2 | Full VRAM Offload |
| `llama3.2:3b` | **1763.39** | **63.13** | 2 | Full VRAM Offload |
| `llama3.1:8b` | **169.44** | **8.18** | 2 | Hybrid Offload (VRAM + System RAM) |
