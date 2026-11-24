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

When a model fits entirely into the GTX 1650's 4 GB VRAM, the results are remarkably capable. Llama 3.2 1B processes input prompts at over 90 tokens per second and streams responses at roughly 45 to 55 tokens per second. That is faster than standard reading speed. Bumping up to Llama 3.2 3B—which offers substantially better reasoning for coding and technical drafting—the system still maintains 32 to 38 tokens per second in generation. For day-to-day coding assistance, summarizing articles, and interactive terminal chat, a 3B model running fully on a 4 GB card feels instantaneous.

The picture changes dramatically when we step up to an 8B model. Because Llama 3.1 8B spills across the PCI Express boundary into system memory, generation throughput drops to approximately 5 to 7 tokens per second. While 6 tokens per second is readable—roughly equivalent to a steady human typing speed—the initial prompt evaluation introduces a noticeable two-to-four-second pause as data shuffles across the bus.

For 14B models running completely on the Intel i5 CPU and DDR4 RAM, generation slows to 1.8 to 2.4 tokens per second. At that pace, interactive conversation feels sluggish, though it remains useful for overnight batch processing tasks where immediacy does not matter.

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
