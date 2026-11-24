#!/usr/bin/env python3
"""
Hardware-Aware LLM Inference Benchmark
Standardized evaluation script for measuring Time-To-First-Token (TTFT)
and generation throughput using local Ollama endpoints.
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def get_cpu_info() -> str:
    """Extract clean CPU model string on Linux, fallback to platform."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_ram_info() -> str:
    """Extract total system RAM in GiB."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    return f"{kb / (1024 * 1024):.1f} GiB"
    except Exception:
        pass
    return "Unknown RAM"


def get_gpu_info() -> Dict[str, str]:
    """Extract NVIDIA GPU details via nvidia-smi if available."""
    info = {
        "model": "No dedicated NVIDIA GPU detected",
        "vram": "N/A",
        "driver": "N/A",
        "cuda": "N/A",
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            parts = [p.strip() for p in out.split(",")]
            if len(parts) >= 3:
                info["model"] = parts[0]
                info["vram"] = f"{int(parts[1]) / 1024:.1f} GiB ({parts[1]} MiB)"
                info["driver"] = parts[2]

        cuda_out = subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.DEVNULL
        )
        for line in cuda_out.splitlines():
            if "CUDA Version:" in line:
                info["cuda"] = line.split("CUDA Version:")[1].split()[0].replace("|", "").strip()
                break
    except Exception:
        pass
    return info


def detect_system_profile() -> Dict[str, Any]:
    """Collate complete host hardware and OS profile."""
    gpu = get_gpu_info()
    return {
        "os": platform.platform(),
        "cpu": get_cpu_info(),
        "ram": get_ram_info(),
        "gpu_model": gpu["model"],
        "gpu_vram": gpu["vram"],
        "gpu_driver": gpu["driver"],
        "cuda_version": gpu["cuda"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def parse_ollama_timing(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract prompt processing and token evaluation speed from Ollama response.
    Timestamps in Ollama are given in nanoseconds (1 second = 1e9 ns).
    """
    prompt_count = data.get("prompt_eval_count", 0)
    prompt_ns = data.get("prompt_eval_duration", 0)
    eval_count = data.get("eval_count", 0)
    eval_ns = data.get("eval_duration", 0)
    load_ns = data.get("load_duration", 0)

    prompt_tok_per_sec = (prompt_count / (prompt_ns / 1e9)) if prompt_ns > 0 else 0.0
    eval_tok_per_sec = (eval_count / (eval_ns / 1e9)) if eval_ns > 0 else 0.0
    load_sec = load_ns / 1e9

    return {
        "prompt_tokens": prompt_count,
        "prompt_speed_tok_s": round(prompt_tok_per_sec, 2),
        "eval_tokens": eval_count,
        "eval_speed_tok_s": round(eval_tok_per_sec, 2),
        "load_time_s": round(load_sec, 2),
    }


def send_ollama_request(
    endpoint: str, model: str, prompt: str, num_predict: int = 128
) -> Dict[str, Any]:
    """Execute raw HTTP request to Ollama generate endpoint."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0.0,  # Deterministic / zero-temperature for scientific consistency
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_benchmark_for_model(
    endpoint: str,
    model: str,
    prompt: str,
    runs: int = 3,
    num_predict: int = 128,
) -> Dict[str, Any]:
    """Execute warm-up followed by multiple measurement passes."""
    print(f"\n==================================================")
    print(f"Benchmarking Model: {model}")
    print(f"==================================================")

    print(" -> Warm-up pass (loading weights into memory)...")
    send_ollama_request(endpoint, model, "Warm-up inference.", num_predict=16)

    prompt_speeds = []
    eval_speeds = []

    for i in range(1, runs + 1):
        print(f" -> Iteration {i}/{runs}...")
        raw_res = send_ollama_request(endpoint, model, prompt, num_predict=num_predict)
        metrics = parse_ollama_timing(raw_res)
        prompt_speeds.append(metrics["prompt_speed_tok_s"])
        eval_speeds.append(metrics["eval_speed_tok_s"])
        print(
            f"    Prompt: {metrics['prompt_speed_tok_s']} tok/s | "
            f"Generation: {metrics['eval_speed_tok_s']} tok/s ({metrics['eval_tokens']} tokens)"
        )
        time.sleep(1)

    avg_prompt = round(sum(prompt_speeds) / len(prompt_speeds), 2)
    avg_eval = round(sum(eval_speeds) / len(eval_speeds), 2)

    return {
        "model": model,
        "runs": runs,
        "tokens_generated": num_predict,
        "prompt_speeds": prompt_speeds,
        "eval_speeds": eval_speeds,
        "avg_prompt_speed_tok_s": avg_prompt,
        "avg_eval_speed_tok_s": avg_eval,
    }


def format_markdown_report(profile: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    """Format benchmark results into a clean GitHub markdown document."""
    lines = [
        "## System Hardware Profile",
        "",
        f"- **Processor (CPU):** {profile.get('cpu', 'N/A')}",
        f"- **System Memory (RAM):** {profile.get('ram', 'N/A')}",
        f"- **Dedicated Graphics (GPU):** {profile.get('gpu_model', 'N/A')}",
        f"- **Video Memory (VRAM):** {profile.get('gpu_vram', 'N/A')}",
        f"- **GPU Driver & CUDA:** Driver {profile.get('gpu_driver', 'N/A')} | CUDA {profile.get('cuda_version', 'N/A')}",
        f"- **Operating System:** {profile.get('os', 'N/A')}",
        f"- **Benchmark Date:** {profile.get('timestamp', 'N/A')}",
        "",
        "## Inference Performance Results",
        "",
        "| Model Name | Prompt Processing (tok/s) | Generation Speed (tok/s) | Samples | Evaluation Mode |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ]

    for item in results:
        m = item["model"]
        p_spd = item["avg_prompt_speed_tok_s"]
        e_spd = item["avg_eval_speed_tok_s"]
        runs = item["runs"]
        
        # Categorize execution mode based on model naming / parameters
        if any(tag in m for tag in ["1b", "0.5b", "1.5b"]):
            mode = "Full GPU VRAM Offload"
        elif any(tag in m for tag in ["3b", "mini"]):
            mode = "Full GPU VRAM Offload"
        elif any(tag in m for tag in ["7b", "8b"]):
            mode = "Hybrid Offload (GPU VRAM + System RAM)"
        else:
            mode = "System RAM / CPU Bound"

        lines.append(f"| `{m}` | **{p_spd}** | **{e_spd}** | {runs} | {mode} |")

    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> Prompt processing measures time-to-first-token throughput (input comprehension).")
    lines.append("> Generation speed measures sustained token output rate during active text synthesis.")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Reproducible LLM Inference Benchmark Tool")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"],
        help="List of Ollama models to benchmark",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of iterations per model (default: 3)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=128,
        help="Target number of output tokens generated per pass (default: 128)",
    )
    parser.add_argument(
        "--output-json",
        default="results/benchmark_results.json",
        help="Path to save raw JSON output",
    )
    parser.add_argument(
        "--output-md",
        default="results/benchmark_table.md",
        help="Path to save generated Markdown report",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    profile = detect_system_profile()
    print("Detected System Profile:")
    print(f"  CPU: {profile['cpu']}")
    print(f"  RAM: {profile['ram']}")
    print(f"  GPU: {profile['gpu_model']} ({profile['gpu_vram']})")
    print(f"  OS:  {profile['os']}")

    standard_prompt = (
        "Explain the physical principles of how an airplane wing generates lift, "
        "focusing on pressure differentials and airflow speed. Be concise and thorough."
    )

    results = []
    for model in args.models:
        try:
            res = run_benchmark_for_model(
                args.endpoint,
                model,
                standard_prompt,
                runs=args.runs,
                num_predict=args.tokens,
            )
            results.append(res)
        except Exception as e:
            print(f"[!] Error benchmarking {model}: {e}")

    if results:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        full_payload = {
            "system_profile": profile,
            "benchmark_results": results,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(full_payload, f, indent=2)
        print(f"\n[✓] Saved raw data to {args.output_json}")

        md_content = format_markdown_report(profile, results)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[✓] Saved markdown report to {args.output_md}")
        print("\n" + md_content)


if __name__ == "__main__":
    main()
