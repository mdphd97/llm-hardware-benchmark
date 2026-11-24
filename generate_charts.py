#!/usr/bin/env python3
"""
Zero-Dependency SVG Chart Generator for LLM Hardware Benchmark Suite.
Reads empirical JSON datasets from results/ and builds GitHub-ready vector graphics.
"""

import glob
import html
import json
import os
from typing import Dict


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

  <!-- Decorative Compute Nodes -->
  <g transform="translate(940, 55)" opacity="0.95">
    <rect x="0" y="20" width="84" height="44" rx="8" fill="#1e293b" stroke="#38bdf8" stroke-width="1.5" />
    <text x="42" y="47" text-anchor="middle" fill="#38bdf8" font-family="{MONO_FAMILY}" font-size="13" font-weight="700">CUDA</text>

    <rect x="115" y="0" width="90" height="44" rx="8" fill="#1e293b" stroke="#818cf8" stroke-width="1.5" />
    <text x="160" y="27" text-anchor="middle" fill="#a5b4fc" font-family="{MONO_FAMILY}" font-size="13" font-weight="700">CPU RAM</text>

    <rect x="115" y="65" width="90" height="44" rx="8" fill="#1e293b" stroke="#34d399" stroke-width="1.5" />
    <text x="160" y="92" text-anchor="middle" fill="#6ee7b7" font-family="{MONO_FAMILY}" font-size="13" font-weight="700">VULKAN</text>

    <rect x="20" y="115" width="165" height="40" rx="8" fill="#1e293b" stroke="#f472b6" stroke-width="1.5" />
    <text x="102" y="140" text-anchor="middle" fill="#fbcfe8" font-family="{MONO_FAMILY}" font-size="12" font-weight="700">Local Ollama API</text>

    <path d="M 84 42 L 115 22 M 84 42 L 115 87 M 102 115 L 160 109" stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 3" />
  </g>

  <!-- Title & Tagline -->
  <text x="64" y="95" fill="#f8fafc" font-family="{FONT_FAMILY}" font-size="38" font-weight="800">
    LLM Hardware Performance Benchmark
  </text>
  <rect x="64" y="112" width="140" height="4" rx="2" fill="url(#accentGrad)" />
  <text x="64" y="152" fill="#cbd5e1" font-family="{FONT_FAMILY}" font-size="18" font-weight="400">
    Empirical Inference Throughput &amp; Memory Topology Across Consumer Compute Tiers
  </text>

  <!-- Tech Pills -->
  <g transform="translate(64, 186)">
    <rect x="0" y="0" width="96" height="32" rx="16" fill="#0f172a" stroke="#38bdf8" stroke-width="1.2" />
    <text x="48" y="21" text-anchor="middle" fill="#7dd3fc" font-family="{MONO_FAMILY}" font-size="12" font-weight="600">NVIDIA CUDA</text>

    <rect x="110" y="0" width="114" height="32" rx="16" fill="#0f172a" stroke="#818cf8" stroke-width="1.2" />
    <text x="167" y="21" text-anchor="middle" fill="#c7d2fe" font-family="{MONO_FAMILY}" font-size="12" font-weight="600">CPU AVX-512</text>

    <rect x="238" y="0" width="126" height="32" rx="16" fill="#0f172a" stroke="#34d399" stroke-width="1.2" />
    <text x="301" y="21" text-anchor="middle" fill="#6ee7b7" font-family="{MONO_FAMILY}" font-size="12" font-weight="600">Vulkan Iris Xe</text>

    <rect x="378" y="0" width="128" height="32" rx="16" fill="#0f172a" stroke="#f59e0b" stroke-width="1.2" />
    <text x="442" y="21" text-anchor="middle" fill="#fde68a" font-family="{MONO_FAMILY}" font-size="12" font-weight="600">System DDR4</text>

    <rect x="520" y="0" width="142" height="32" rx="16" fill="#0f172a" stroke="#f472b6" stroke-width="1.2" />
    <text x="591" y="21" text-anchor="middle" fill="#fbcfe8" font-family="{MONO_FAMILY}" font-size="12" font-weight="600">Zero-Dep Python</text>
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

    lx = 40
    ly = 95
    for _, label, color in tiers:
        parts.append(f'  <rect x="{lx}" y="{ly - 11}" width="14" height="14" rx="3" fill="{color}" />')
        parts.append(f'  <text x="{lx + 22}" y="{ly}" fill="#cbd5e1" font-family="{FONT_FAMILY}" font-size="12.5">{html.escape(label)}</text>')
        lx += len(label) * 6.8 + 42

    for tick in [0, 25, 50, 75, 100]:
        if tick > max_val:
            continue
        gx = left_margin + (tick / max_val) * chart_width
        parts.append(f'  <line x1="{gx:.1f}" y1="{top_pad}" x2="{gx:.1f}" y2="{height - bottom_pad}" stroke="#334155" stroke-width="1" stroke-opacity="0.5" stroke-dasharray="4 4" />')
        parts.append(f'  <text x="{gx:.1f}" y="{height - bottom_pad + 22}" text-anchor="middle" fill="#64748b" font-family="{MONO_FAMILY}" font-size="12">{tick} tok/s</text>')

    for idx, model in enumerate(models):
        gy = top_pad + idx * group_height
        if idx > 0:
            parts.append(f'  <line x1="40" y1="{gy - 10}" x2="{width - 40}" y2="{gy - 10}" stroke="#1e293b" stroke-width="1" />')

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


def main():
    os.makedirs("assets", exist_ok=True)
    generate_hero_banner("assets/hero_banner.svg")
    generate_throughput_chart("results", "assets/compute_tier_comparison.svg")
    print("[✓] Generated vector charts in assets/")


if __name__ == "__main__":
    main()
