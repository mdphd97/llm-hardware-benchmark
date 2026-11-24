#!/usr/bin/env python3
"""
Zero-Dependency SVG Chart Generator for LLM Hardware Benchmark Suite.
Reads empirical JSON datasets from results/ and builds GitHub-ready vector graphics.
"""

import glob
import html
import json
import os
from typing import Any, Dict, List


FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO_FAMILY = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"


def generate_hero_banner(output_path: str) -> None:
    """Create a sleek dark-slate technical header banner SVG."""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 280" width="100%" height="280">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0b1120" />
      <stop offset="55%" stop-color="#111827" />
      <stop offset="100%" stop-color="#0f172a" />
    </linearGradient>
    <linearGradient id="accentGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38bdf8" />
      <stop offset="50%" stop-color="#818cf8" />
      <stop offset="100%" stop-color="#34d399" />
    </linearGradient>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="1" />
    </pattern>
  </defs>

  <rect width="1200" height="280" rx="16" fill="#0f172a" />
  <rect width="1200" height="280" rx="16" fill="url(#bgGrad)" opacity="0.9" />
  <rect width="1200" height="280" rx="16" fill="url(#grid)" opacity="0.5" />
  <rect x="0" y="0" width="1200" height="4" rx="2" fill="url(#accentGrad)" />

  <!-- Decorative Hardware Nodes -->
  <g transform="translate(940, 55)" opacity="0.95">
    <rect x="0" y="20" width="76" height="44" rx="8" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5" />
    <text x="38" y="47" text-anchor="middle" fill="#38bdf8" font-family="{MONO_FAMILY}" font-size="13" font-weight="700">CUDA</text>

    <rect x="115" y="0" width="84" height="44" rx="8" fill="#1e293b" stroke="#818cf8" stroke-width="1.5" />
    <text x="157" y="27" text-anchor="middle" fill="#a5b4fc" font-family="{MONO_FAMILY}" font-size="13" font-weight="700">CPU RAM</text>

    <rect x="115" y="65" width="84" height="44" rx="8" fill="#1e293b" stroke="#34d399" stroke-width="1.5" />
    <text x="157" y="92" text-anchor="middle" fill="#6ee7b7" font-family="{MONO_FAMILY}" font-size="13" font-weight="700">VULKAN</text>

    <rect x="20" y="115" width="160" height="40" rx="8" fill="#1e293b" stroke="#f59e0b" stroke-width="1.5" />
    <text x="100" y="140" text-anchor="middle" fill="#fcd34d" font-family="{MONO_FAMILY}" font-size="12" font-weight="700">LoRa + Qdrant MCP</text>

    <path d="M 76 42 L 115 22 M 76 42 L 115 87 M 100 115 L 157 109" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3" />
  </g>

  <!-- Title & Tagline -->
  <text x="64" y="95" fill="#f8fafc" font-family="{FONT_FAMILY}" font-size="38" font-weight="800">
    LLM Hardware &amp; Edge Agent Benchmark
  </text>
  <rect x="64" y="112" width="140" height="4" rx="2" fill="url(#accentGrad)" />
  <text x="64" y="152" fill="#cbd5e1" font-family="{FONT_FAMILY}" font-size="18" font-weight="400">
    Empirical Inference Throughput, Memory Topology &amp; LoRa-to-LLM Pipeline Telemetry
  </text>

  <!-- Tech Pills -->
  <g transform="translate(64, 186)">
    <rect x="0" y="0" width="94" height="32" rx="16" fill="#0f172a" stroke="#38bdf8" stroke-width="1.2" />
    <text x="47" y="21" text-anchor="middle" fill="#7dd3fc" font-family="{MONO_FAMILY}" font-size="12" font-weight="600">NVIDIA CUDA</text>

    <rect x="108" y="0" width="112" height="32" rx="16" fill="#0f172a" stroke="#818cf8" stroke-width="1.2" />
    <text x="164" y="21" text-anchor="middle" fill="#c7d2fe" font-family="{MONO_FAMILY}" font-size="12" font-weight="600">CPU AVX-512</text>

    <rect x="234" y="0" width="124" height="32" rx="16" fill="#0f172a" stroke="#34d399" stroke-width="1.2" />
    <text x="296" y="21" text-anchor="middle" fill="#6ee7b7" font-family="{MONO_FAMILY}" font-size="12" font-weight="600">Vulkan Iris Xe</text>

    <rect x="372" y="0" width="142" height="32" rx="16" fill="#0f172a" stroke="#f59e0b" stroke-width="1.2" />
    <text x="443" y="21" text-anchor="middle" fill="#fde68a" font-family="{MONO_FAMILY}" font-size="12" font-weight="600">LoRa + Qdrant DB</text>

    <rect x="528" y="0" width="142" height="32" rx="16" fill="#0f172a" stroke="#f472b6" stroke-width="1.2" />
    <text x="599" y="21" text-anchor="middle" fill="#fbcfe8" font-family="{MONO_FAMILY}" font-size="12" font-weight="600">Zero-Dep Python</text>
  </g>
</svg>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)


def load_benchmark_datasets(results_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load results keyed by [model][system_id] -> metrics."""
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "benchmark_results.json"))):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        sys_id = payload.get("system_profile", {}).get("system_id", "unknown")
        for item in payload.get("benchmark_results", []):
            model = item["model"]
            data.setdefault(model, {})[sys_id] = {
                "gen": item["avg_eval_speed_tok_s"],
                "prompt": item["avg_prompt_speed_tok_s"],
                "mode": item.get("execution_mode", ""),
            }
    return data


def generate_throughput_chart(results_dir: str, output_path: str) -> None:
    """Create grouped horizontal bar chart comparing generation speed across compute tiers."""
    datasets = load_benchmark_datasets(results_dir)
    models = ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"]
    # Filter to models present in datasets, keep canonical order
    models = [m for m in models if m in datasets] + [m for m in sorted(datasets.keys()) if m not in models]

    tiers = [
        ("gtx_1650_i5_11300h", "NVIDIA GTX 1650 (4 GB VRAM CUDA / Hybrid)", "#38bdf8"),
        ("cpu_ram_i5_11300h_310ghz", "Intel i5-11300H CPU + DDR4 RAM (0 GPU layers)", "#818cf8"),
        ("intel_iris_xe_vulkan", "Intel Iris Xe iGPU (Vulkan Unified RAM)", "#34d399"),
    ]

    max_val = 115.0
    for m in models:
        for sys_id, _, _ in tiers:
            if sys_id in datasets[m]:
                max_val = max(max_val, datasets[m][sys_id]["gen"] * 1.12)

    width = 1100
    group_height = 148
    top_pad = 125
    bottom_pad = 50
    height = top_pad + len(models) * group_height + bottom_pad

    left_margin = 240
    chart_width = 740

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}">',
        f'  <rect width="{width}" height="{height}" rx="14" fill="#0f172a" stroke="#1e293b" stroke-width="1.5" />',
        f'  <text x="40" y="44" fill="#f8fafc" font-family="{FONT_FAMILY}" font-size="22" font-weight="700">Token Generation Throughput Across Compute Tiers (tokens/sec)</text>',
        f'  <text x="40" y="70" fill="#94a3b8" font-family="{FONT_FAMILY}" font-size="14">Higher is better • Zero-temperature greedy decoding (128 tokens) • Single-laptop hardware comparison</text>',
    ]

    # Legend
    lx = 40
    ly = 95
    for _, label, color in tiers:
        parts.append(f'  <rect x="{lx}" y="{ly - 11}" width="14" height="14" rx="3" fill="{color}" />')
        parts.append(f'  <text x="{lx + 22}" y="{ly}" fill="#cbd5e1" font-family="{FONT_FAMILY}" font-size="12.5">{html.escape(label)}</text>')
        lx += len(label) * 6.8 + 42

    # Grid lines
    for tick in [0, 25, 50, 75, 100]:
        if tick > max_val:
            continue
        gx = left_margin + (tick / max_val) * chart_width
        parts.append(f'  <line x1="{gx:.1f}" y1="{top_pad}" x2="{gx:.1f}" y2="{height - bottom_pad}" stroke="#334155" stroke-width="1" stroke-opacity="0.5" stroke-dasharray="4 4" />')
        parts.append(f'  <text x="{gx:.1f}" y="{height - bottom_pad + 22}" text-anchor="middle" fill="#64748b" font-family="{MONO_FAMILY}" font-size="12">{tick} tok/s</text>')

    # Bars per model
    for idx, model in enumerate(models):
        gy = top_pad + idx * group_height
        if idx > 0:
            parts.append(f'  <line x1="40" y1="{gy - 10}" x2="{width - 40}" y2="{gy - 10}" stroke="#1e293b" stroke-width="1" />')

        # Model Label
        parts.append(f'  <text x="40" y="{gy + 46}" fill="#f1f5f9" font-family="{MONO_FAMILY}" font-size="16" font-weight="700">{html.escape(model)}</text>')
        sub_note = "Fits in 4 GB VRAM" if ("1b" in model or "3b" in model) else "Exceeds 4 GB VRAM (Hybrid)"
        parts.append(f'  <text x="40" y="{gy + 68}" fill="#64748b" font-family="{FONT_FAMILY}" font-size="12">{sub_note}</text>')

        bar_y = gy + 10
        for sys_id, _, color in tiers:
            val = datasets.get(model, {}).get(sys_id, {}).get("gen", 0.0)
            bw = max((val / max_val) * chart_width, 4.0) if val > 0 else 0
            parts.append(f'  <rect x="{left_margin}" y="{bar_y}" width="{chart_width}" height="28" rx="5" fill="#1e293b" opacity="0.45" />')
            if bw > 0:
                parts.append(f'  <rect x="{left_margin}" y="{bar_y}" width="{bw:.1f}" height="28" rx="5" fill="{color}" />')
                parts.append(f'  <text x="{left_margin + bw + 10:.1f}" y="{bar_y + 19}" fill="#f8fafc" font-family="{MONO_FAMILY}" font-size="13" font-weight="700">{val:.2f} tok/s</text>')
            bar_y += 38

    parts.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def generate_lora_waterfall_chart(stress_json_path: str, output_path: str) -> None:
    """Create Sensor-to-Agent Latency Waterfall SVG from stress test JSON."""
    if not os.path.exists(stress_json_path):
        return

    with open(stress_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    wf = data["latency_waterfall_ms"]
    en = data["energy_profile"]
    tp = data["throughput"]

    stages = [
        ("1. LoRa Radio Airtime (SX1262 SF7)", wf["avg_lora_toa_ms"], "#38bdf8"),
        ("2. Ollama Embedding (nomic-embed-text)", wf["avg_embedding_ms"], "#818cf8"),
        ("3. Qdrant Vector Search (Docker)", wf["avg_qdrant_search_ms"], "#34d399"),
        ("4. LLM MCP Reasoning (llama3.2:1b)", wf["avg_llm_mcp_reason_ms"], "#f59e0b"),
    ]
    total_ms = wf["total_anomaly_e2e_ms"]

    width = 1100
    height = 370
    left_margin = 365
    bar_max_w = 555

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="{height}">',
        f'  <defs><clipPath id="stackClip"><rect x="40" y="95" width="1020" height="24" rx="6" /></clipPath></defs>',
        f'  <rect width="{width}" height="{height}" rx="14" fill="#0f172a" stroke="#1e293b" stroke-width="1.5" />',
        f'  <text x="40" y="44" fill="#f8fafc" font-family="{FONT_FAMILY}" font-size="22" font-weight="700">LoRa-to-Local-LLM Sensor-to-Agent Latency Waterfall</text>',
        f'  <text x="40" y="70" fill="#94a3b8" font-family="{FONT_FAMILY}" font-size="14">End-to-End Physical Anomaly Triage: {tp["packets_ingested"]} LoRa Nodes ({tp["anomalies_triaged"]} anomalies) • Total E2E: {total_ms:.2f} ms</text>',
    ]

    # Stacked proportional overview bar
    stack_y = 95
    curr_x = 40.0
    stack_total_w = 1020.0
    parts.append('  <g clip-path="url(#stackClip)">')
    for label, ms, color in stages:
        seg_w = max((ms / total_ms) * stack_total_w, 4.0)
        parts.append(f'    <rect x="{curr_x:.1f}" y="{stack_y}" width="{seg_w:.1f}" height="24" fill="{color}" />')
        curr_x += seg_w
    parts.append('  </g>')

    # Individual Waterfall Rows
    row_y = 148
    for label, ms, color in stages:
        pct = (ms / total_ms) * 100.0
        bw = max((ms / total_ms) * bar_max_w, 6.0)
        parts.append(f'  <text x="40" y="{row_y + 18}" fill="#e2e8f0" font-family="{FONT_FAMILY}" font-size="14" font-weight="600">{html.escape(label)}</text>')
        parts.append(f'  <rect x="{left_margin}" y="{row_y}" width="{bar_max_w}" height="26" rx="5" fill="#1e293b" opacity="0.45" />')
        parts.append(f'  <rect x="{left_margin}" y="{row_y}" width="{bw:.1f}" height="26" rx="5" fill="{color}" />')
        parts.append(f'  <text x="{left_margin + bw + 12:.1f}" y="{row_y + 18}" fill="#f8fafc" font-family="{MONO_FAMILY}" font-size="13" font-weight="700">{ms:.2f} ms ({pct:.1f}%)</text>')
        row_y += 42

    # Footer Power Summary Strip
    parts.append(f'  <rect x="40" y="320" width="1020" height="32" rx="6" fill="#1e293b" />')
    parts.append(
        f'  <text x="550" y="341" text-anchor="middle" fill="#cbd5e1" font-family="{MONO_FAMILY}" font-size="13">'
        f'Host GPU Power: {en["avg_gpu_watts"]} W avg  |  Total Energy: {en["gpu_energy_joules"]} J  |  Peak Temp: {en["peak_gpu_temp_c"]} C  |  Qdrant Upsert: {wf["avg_qdrant_upsert_ms"]} ms'
        f'</text>'
    )
    parts.append("</svg>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def generate_heltec_architecture_svg(output_path: str) -> None:
    """Create Heltec WiFi LoRa 32 V3 End-to-End Edge-to-LLM Architecture Diagram."""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 320" width="100%" height="320">
  <rect width="1100" height="320" rx="14" fill="#0f172a" stroke="#1e293b" stroke-width="1.5" />
  <text x="40" y="42" fill="#f8fafc" font-family="{FONT_FAMILY}" font-size="21" font-weight="700">Heltec WiFi LoRa 32 V3: Edge TinyML to Local LLM Agent Architecture</text>
  <text x="40" y="66" fill="#94a3b8" font-family="{FONT_FAMILY}" font-size="13.5">Ready-to-flash PlatformIO dual-role firmware paired with Linux Docker Qdrant &amp; Ollama MCP</text>

  <!-- Node 1: Heltec Edge -->
  <rect x="40" y="96" width="230" height="156" rx="10" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5" />
  <text x="155" y="122" text-anchor="middle" fill="#38bdf8" font-family="{MONO_FAMILY}" font-size="13.5" font-weight="700">Heltec V3 Edge Node</text>
  <text x="155" y="146" text-anchor="middle" fill="#e2e8f0" font-family="{FONT_FAMILY}" font-size="12">ESP32-S3FN8 + SSD1306</text>
  <text x="155" y="170" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• INT8 TinyML Scoring</text>
  <text x="155" y="190" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• SX1262 LoRa 915MHz SF7</text>
  <text x="155" y="210" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• Vext Cut + Deep Sleep</text>
  <text x="155" y="234" text-anchor="middle" fill="#7dd3fc" font-family="{MONO_FAMILY}" font-size="11">GPIO 8/12/13/14 • ~61.7ms ToA</text>

  <!-- Arrow 1->2 -->
  <line x1="270" y1="174" x2="320" y2="174" stroke="#38bdf8" stroke-width="2" stroke-dasharray="5 3" />
  <polygon points="320,169 330,174 320,179" fill="#38bdf8" />
  <text x="300" y="162" text-anchor="middle" fill="#7dd3fc" font-family="{MONO_FAMILY}" font-size="10.5">RF</text>

  <!-- Node 2: Heltec Gateway -->
  <rect x="330" y="96" width="210" height="156" rx="10" fill="#1e293b" stroke="#818cf8" stroke-width="1.5" />
  <text x="435" y="122" text-anchor="middle" fill="#a5b4fc" font-family="{MONO_FAMILY}" font-size="13.5" font-weight="700">Heltec V3 Gateway</text>
  <text x="435" y="146" text-anchor="middle" fill="#e2e8f0" font-family="{FONT_FAMILY}" font-size="12">SX1262 RX + USB-C UART</text>
  <text x="435" y="170" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• Appends RSSI &amp; SNR</text>
  <text x="435" y="190" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• OLED Live Telemetry</text>
  <text x="435" y="210" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• 115200 Baud JSON Stream</text>
  <text x="435" y="234" text-anchor="middle" fill="#c7d2fe" font-family="{MONO_FAMILY}" font-size="11">--serial-port /dev/ttyUSB0</text>

  <!-- Arrow 2->3 -->
  <line x1="540" y1="174" x2="590" y2="174" stroke="#818cf8" stroke-width="2" />
  <polygon points="590,169 600,174 590,179" fill="#818cf8" />
  <text x="570" y="162" text-anchor="middle" fill="#c7d2fe" font-family="{MONO_FAMILY}" font-size="10.5">USB</text>

  <!-- Node 3: Docker Qdrant + Embed -->
  <rect x="600" y="96" width="220" height="156" rx="10" fill="#1e293b" stroke="#34d399" stroke-width="1.5" />
  <text x="710" y="122" text-anchor="middle" fill="#34d399" font-family="{MONO_FAMILY}" font-size="13.5" font-weight="700">Docker Vector Memory</text>
  <text x="710" y="146" text-anchor="middle" fill="#e2e8f0" font-family="{FONT_FAMILY}" font-size="12">Qdrant + nomic-embed</text>
  <text x="710" y="170" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• 768-dim Embeddings</text>
  <text x="710" y="190" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• Cosine Context Lookup</text>
  <text x="710" y="210" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• 1.78 ms Vector Search</text>
  <text x="710" y="234" text-anchor="middle" fill="#6ee7b7" font-family="{MONO_FAMILY}" font-size="11">Port 6333 • Local Linux</text>

  <!-- Arrow 3->4 -->
  <line x1="820" y1="174" x2="860" y2="174" stroke="#f59e0b" stroke-width="2" />
  <polygon points="860,169 870,174 860,179" fill="#f59e0b" />

  <!-- Node 4: Local LLM MCP -->
  <rect x="870" y="96" width="190" height="156" rx="10" fill="#1e293b" stroke="#f59e0b" stroke-width="1.5" />
  <text x="965" y="122" text-anchor="middle" fill="#fbbf24" font-family="{MONO_FAMILY}" font-size="13.5" font-weight="700">Local LLM Agent</text>
  <text x="965" y="146" text-anchor="middle" fill="#e2e8f0" font-family="{FONT_FAMILY}" font-size="12">Ollama + MCP Dispatch</text>
  <text x="965" y="170" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• Anomaly Triage</text>
  <text x="965" y="190" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• JSON Tool Selection</text>
  <text x="965" y="210" text-anchor="middle" fill="#94a3b8" font-family="{MONO_FAMILY}" font-size="11.5">• 1,012 ms Inference</text>
  <text x="965" y="234" text-anchor="middle" fill="#fde68a" font-family="{MONO_FAMILY}" font-size="11">1.60 s Total E2E Loop</text>

  <!-- Footer -->
  <rect x="40" y="272" width="1020" height="30" rx="6" fill="#1e293b" />
  <text x="550" y="292" text-anchor="middle" fill="#cbd5e1" font-family="{MONO_FAMILY}" font-size="12.5">
    firmware/heltec_v3_sx1262/  --&gt;  lora-agent-pipeline/agent_bridge.py  --&gt;  Verified Hardware &amp; Latency Telemetry
  </text>
</svg>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)


def main():
    os.makedirs("assets", exist_ok=True)
    generate_hero_banner("assets/hero_banner.svg")
    generate_throughput_chart("results", "assets/compute_tier_comparison.svg")
    generate_lora_waterfall_chart("results/lora_agent_stress_test.json", "assets/lora_latency_waterfall.svg")
    generate_heltec_architecture_svg("assets/heltec_edge_to_llm.svg")
    print("[✓] Generated vector charts in assets/")


if __name__ == "__main__":
    main()
