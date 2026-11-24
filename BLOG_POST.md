# What Can a 4 GB Graphics Card Actually Run? An Empirical Review of Local LLMs

Can an entry-level laptop with four gigabytes of video memory still run modern artificial intelligence models? In simple terms, yes, but the user experience depends almost entirely on where the model's parameters live during inference.

When we read benchmark reports in the machine learning literature, the focus is almost universally centered on high-end enterprise hardware—clusters of 80 GB H100s or desktop rigs running dual RTX 4090s. While those evaluations tell us what is theoretically possible at scale, they offer little practical guidance to the millions of software engineers, researchers, and students who develop on modest personal computers. To bridge this gap, we conducted an empirical benchmark of modern open-weight language models on an accessible machine: an Intel Core i5-11300H processor, 24 GB of system RAM, and an NVIDIA GeForce GTX 1650 with 4 GB of VRAM.

Before we evaluate the performance data, we should clarify how an inference engine actually divides work across personal computer hardware.

---

### The Mechanical Reality of Autoregressive Inference

What is token generation in simple physical terms? During inference, a language model generates text one word fragment (token) at a time. For every single token produced, the computer's memory subsystem must read the entire model weight file from memory into compute cores, calculate the next probability distribution, and append the winning token to the prompt history.

Computer science research into autoregressive transformer inference (Shazeer, 2019) demonstrated early on that generation speed is fundamentally memory-bandwidth bound rather than arithmetic-bound. To understand why this matters for a 4 GB graphics card, we can use a workshop assembly line as an analogy.

Think of the graphics card's high-speed video memory (GDDR6) as an assembly table right in front of a technician. The technician can pick up parts from the table at blistering speed—roughly 192 gigabytes per second on a GTX 1650. In contrast, the computer's system memory (DDR4) is like a storage rack in the back of the building, providing roughly 40 gigabytes per second. Connecting the two is the PCI Express bus, which functions like a narrow hallway. 

If an entire small model fits directly on the table in front of the technician, work moves smoothly. But if a model is too large, the technician must walk down the hallway to fetch parts from the back room for every single screw they tighten. That physical round trip is where performance collapses.

---

### The Evaluation Matrix

We structured our test across three operational categories that reflect real-world usage patterns:

1. **Lightweight Models (1 Billion to 3 Billion Parameters):** We tested Llama 3.2 1B and Llama 3.2 3B quantized at 4-bit precision (Q4_K_M). These models require approximately 1.3 GB to 2.2 GB of memory, fitting entirely within our 4 GB video memory ceiling with room left over for the context cache.
2. **Standard Workhorse Models (7 Billion to 8 Billion Parameters):** We tested Llama 3.1 8B (Q4_K_M), which requires roughly 4.7 GB of baseline weight storage. Because this slightly exceeds our 4 GB frame buffer, the inference engine splits the neural network layers: approximately 16 layers remain on the graphics card, while the remaining layers spill over into system DDR4 memory.
3. **Large Desktop Models (14 Billion Parameters):** We examined models like Qwen 2.5 14B, which require 9 to 10 GB of space and run almost entirely on the CPU and system memory.

To eliminate random variance, every test used a zero-temperature greedy decoding pass across a fixed 128-token output window after an initial warm-up run.

---

### What the Numbers Show

Talking about how model scale affects throughput, first we should understand what the primary numbers mean:

- **Prompt Evaluation (Time-to-First-Token):** How quickly the computer ingests the user's instructions.
- **Generation Speed:** How quickly the system streams words back to the screen.

When a model fits entirely into the GTX 1650's 4 GB VRAM, the results exceed expectations. On our test rig, Llama 3.2 1B sustained an average generation rate of **105.03 tokens per second** via CUDA. That is roughly three times faster than comfortable human reading speed. Stepping up to Llama 3.2 3B—which offers substantially stronger reasoning for technical drafting and coding—the GPU comfortably maintained **63.13 tokens per second**. For day-to-day coding assistance, summarizing articles, and interactive terminal chat, a 3B model running fully in VRAM feels instantaneous.

The picture shifts dramatically when we bypass the discrete GPU or exceed its 4 GB boundary. Forcing Llama 3.2 1B and 3B to run strictly on the Intel Core i5-11300H CPU and DDR4 system RAM yields **21.80 tokens per second** and **17.02 tokens per second**, respectively—still readable, yet roughly four to five times slower than the GTX 1650. When we step up to Llama 3.1 8B, hybrid GPU-CPU offloading reaches **8.18 tokens per second**, slightly ahead of pure CPU execution at **6.29 tokens per second**.

Interestingly, we also tested the laptop's integrated Intel Iris Xe Graphics via Ollama's Vulkan backend (`OLLAMA_VULKAN=1` and `OLLAMA_IGPU_ENABLE=1`). Even though the integrated GPU has access to 15.4 GiB of unified system memory, it delivered only **6.96 tokens per second** on 1B, **5.14** on 3B, and **3.34** on 8B. Because both the CPU and integrated GPU share the exact same DDR4 memory channel, the CPU's dedicated AVX-512 and VNNI vector instructions outperform generic Vulkan compute shaders by nearly three to one. That physical constraint explains why modern inference engines disable integrated GPUs by default.

---

### Practical Recommendations for Builders

Along with the benchmarks discussed above, there are notable operational takeaways for anyone setting up a budget workstation:

1. **Prioritize 3B Models for Interactive Tools:** If you are building personal command-line agents, local grammar checkers, or retrieval-augmented generation (RAG) pipelines that require conversational responses, 3-billion-parameter models (such as Llama 3.2 3B or Qwen 2.5 3B) represent the sweet spot for a 4 GB graphics card. They stay entirely within video memory and deliver sub-second responses.
2. **Use Hybrid 8B Offloading for Deep Reasoning:** If your task requires complex architectural planning or nuanced code debugging where a 3B model stumbles, accept the 6-tokens-per-second pace of a hybrid 8B model. It will not feel like a snappy chat interface, but the reasoning accuracy will be noticeably higher.
3. **Invest in System RAM Capacity:** If you must use budget hardware, having 24 GB or 32 GB of system RAM ensures you never face out-of-memory crashes when working with hybrid models or running background batch jobs alongside your desktop applications.

---

### Replicating the Results

Science requires verification. We published the complete automated benchmark script on GitHub along with our raw timing datasets. Anyone running Ollama on their own machine can pull the repository and run the test with a single command:

```bash
git clone https://github.com/mdphd97/llm-hardware-benchmark.git
cd llm-hardware-benchmark
python3 benchmark.py --models llama3.2:1b llama3.2:3b llama3.1:8b
```

By comparing these numbers across different memory configurations, we hope to give developers a realistic, empirical view of what budget hardware can accomplish in the era of open-weight artificial intelligence.
