# Cross-System Hardware Leaderboard

Comparative Large Language Model inference throughput across tested systems and memory tiers.

| System ID | Compute / Memory Tier | Model | Prompt (tok/s) | Generation (tok/s) |
| :--- | :--- | :--- | :---: | :---: |
| `cpu_ram_i5_11300h_310ghz` | CPU + System RAM (0 GPU layers) | `llama3.2:1b` | **701.53** | **21.8** |
| `cpu_ram_i5_11300h_310ghz` | CPU + System RAM (0 GPU layers) | `llama3.2:3b` | **488.92** | **17.02** |
| `cpu_ram_i5_11300h_310ghz` | CPU + System RAM (0 GPU layers) | `llama3.1:8b` | **136.69** | **6.29** |
| `gtx_1650_i5_11300h` | Full VRAM Offload | `llama3.2:1b` | **3121.19** | **105.03** |
| `gtx_1650_i5_11300h` | Full VRAM Offload | `llama3.2:3b` | **1763.39** | **63.13** |
| `gtx_1650_i5_11300h` | Hybrid Offload (VRAM + System RAM) | `llama3.1:8b` | **169.44** | **8.18** |
| `intel_iris_xe_vulkan` | Intel Iris Xe (Vulkan iGPU) | `llama3.2:1b` | **302.92** | **6.96** |
| `intel_iris_xe_vulkan` | Intel Iris Xe (Vulkan iGPU) | `llama3.2:3b` | **199.48** | **5.14** |
| `intel_iris_xe_vulkan` | Intel Iris Xe (Vulkan iGPU) | `llama3.1:8b` | **78.28** | **3.34** |

> To test a new system or CPU/RAM tier, see replication instructions in README.md.
