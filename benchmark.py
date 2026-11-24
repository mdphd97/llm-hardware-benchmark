#!/usr/bin/env python3
"""
Multi-System Hardware-Aware LLM Inference Benchmark
Evaluates and aggregates Large Language Model (LLM) inference performance
across multiple computers, GPU architectures, and memory topologies.
"""

import argparse
import datetime
import glob
import json
import os
import platform
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def sanitize_slug(text: str) -> str:
    """Convert hardware names into clean directory-safe identifiers."""
    cleaned = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "_", cleaned)


def get_cpu_info() -> str:
    """Extract clean CPU model string across Linux and macOS."""
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            return "Apple Silicon / Mac CPU"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_ram_info() -> str:
    """Extract total system RAM across Linux and macOS."""
    if platform.system() == "Darwin":
        try:
            bytes_mem = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            )
            return f"{bytes_mem / (1024**3):.1f} GiB"
        except Exception:
            pass
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
    """Extract GPU details (NVIDIA nvidia-smi or macOS Metal)."""
    info = {
        "model": "CPU / Integrated Graphics",
        "vram": "Shared System RAM",
        "driver": "N/A",
        "cuda": "N/A",
    }
    if platform.system() == "Darwin":
        info["model"] = "Apple Silicon GPU (Metal)"
        info["vram"] = "Unified Memory"
        return info

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
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            if len(parts) >= 3:
                info["model"] = parts[0]
                info["vram"] = f"{int(parts[1]) / 1024:.1f} GiB ({parts[1]} MiB)"
                info["driver"] = parts[2]

        cuda_out = subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.DEVNULL
        )
        for line in cuda_out.splitlines():
            if "CUDA Version:" in line:
                info["cuda"] = (
                    line.split("CUDA Version:")[1].split()[0].replace("|", "").strip()
                )
                break
    except Exception:
        pass
    return info


def detect_system_profile(custom_name: Optional[str] = None) -> Dict[str, Any]:
    """Collate complete host hardware, OS, and slug identifier."""
    gpu = get_gpu_info()
    cpu = get_cpu_info()
    ram = get_ram_info()
    os_desc = platform.platform()

    if custom_name:
        slug = sanitize_slug(custom_name)
    else:
        # Build concise identifier: e.g. gtx_1650-i5_11300h
        gpu_short = gpu["model"].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        cpu_short = cpu.replace("Intel(R) Core(TM) ", "").replace("11th Gen ", "")
        slug = sanitize_slug(f"{gpu_short}_{cpu_short}")

    return {
        "system_id": slug,
        "os": os_desc,
        "cpu": cpu,
        "ram": ram,
        "gpu_model": gpu["model"],
        "gpu_vram": gpu["vram"],
        "gpu_driver": gpu["driver"],
        "cuda_version": gpu["cuda"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def parse_ollama_timing(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract prompt processing and token evaluation speed from Ollama response."""
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
            "temperature": 0.0,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pull_ollama_model(model: str) -> bool:
    """Pull model using Ollama CLI if missing."""
    print(f"[*] Pulling '{model}' via Ollama...")
    ret = subprocess.call(["ollama", "pull", model])
    return ret == 0


def delete_ollama_model(model: str) -> None:
    """Delete model from local Ollama storage to reclaim disk space."""
    print(f"[*] Removing '{model}' from Ollama storage...")
    subprocess.call(["ollama", "rm", model], stdout=subprocess.DEVNULL)


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


def format_system_markdown_table(profile: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    """Generate Markdown report for a specific hardware system."""
    lines = [
        f"### System: `{profile.get('system_id', 'Unknown')}`",
        "",
        f"- **Processor (CPU):** {profile.get('cpu', 'N/A')}",
        f"- **System Memory (RAM):** {profile.get('ram', 'N/A')}",
        f"- **Graphics (GPU):** {profile.get('gpu_model', 'N/A')} ({profile.get('gpu_vram', 'N/A')})",
        f"- **OS:** {profile.get('os', 'N/A')}",
        f"- **Benchmark Date:** {profile.get('timestamp', 'N/A')}",
        "",
        "| Model Name | Prompt Processing (tok/s) | Generation Speed (tok/s) | Runs | Execution Mode |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ]

    for item in results:
        m = item["model"]
        p_spd = item["avg_prompt_speed_tok_s"]
        e_spd = item["avg_eval_speed_tok_s"]
        runs = item["runs"]

        if any(tag in m for tag in ["1b", "0.5b", "1.5b", "3b", "mini"]):
            mode = "Full VRAM Offload"
        elif any(tag in m for tag in ["7b", "8b"]):
            mode = "Hybrid Offload (VRAM + System RAM)"
        else:
            mode = "System RAM / CPU Bound"

        lines.append(f"| `{m}` | **{p_spd}** | **{e_spd}** | {runs} | {mode} |")

    lines.append("")
    return "\n".join(lines)


def aggregate_all_systems(results_root: str = "results") -> str:
    """
    Search results directory for all systems and generate a multi-system
    comparative cross-hardware leaderboard.
    """
    pattern = os.path.join(results_root, "*", "benchmark_results.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return "No benchmark results found across systems."

    lines = [
        "# Cross-System Hardware Leaderboard",
        "",
        "Comparative Large Language Model inference throughput across tested systems.",
        "",
        "| System ID | GPU / Compute | CPU & RAM | Model | Prompt (tok/s) | Generation (tok/s) |",
        "| :--- | :--- | :--- | :--- | :---: | :---: |",
    ]

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            prof = data.get("system_profile", {})
            benchmarks = data.get("benchmark_results", [])

            sys_id = prof.get("system_id", "Unknown")
            gpu = prof.get("gpu_model", "N/A").replace("NVIDIA GeForce ", "")
            vram = prof.get("gpu_vram", "")
            gpu_desc = f"{gpu} ({vram})" if vram and vram != "N/A" else gpu
            cpu_ram = f"{prof.get('cpu', 'N/A')} ({prof.get('ram', 'N/A')})"

            for b in benchmarks:
                m = b.get("model", "N/A")
                p = b.get("avg_prompt_speed_tok_s", 0.0)
                e = b.get("avg_eval_speed_tok_s", 0.0)
                lines.append(f"| `{sys_id}` | {gpu_desc} | {cpu_ram} | `{m}` | **{p}** | **{e}** |")
        except Exception:
            continue

    lines.append("")
    lines.append("> To test a new system and add it to this leaderboard, see replication instructions below.")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-System LLM Inference Benchmark & Aggregation Suite"
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama3.2:1b", "llama3.2:3b"],
        help="List of Ollama models to benchmark",
    )
    parser.add_argument(
        "--system-name",
        default=None,
        help="Custom system name / slug (e.g. 'm2-macbook-pro' or 'rtx-3060-desktop')",
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
        "--delete-after",
        action="store_true",
        help="Automatically remove downloaded models from Ollama after benchmarking to save disk space",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root directory for multi-system results storage",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip benchmark execution and regenerate cross-system comparison table only",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.aggregate_only:
        table = aggregate_all_systems(args.results_dir)
        leaderboard_path = os.path.join(args.results_dir, "LEADERBOARD.md")
        with open(leaderboard_path, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"[✓] Updated leaderboard at {leaderboard_path}")
        print("\n" + table)
        return

    profile = detect_system_profile(args.system_name)
    sys_slug = profile["system_id"]
    sys_dir = os.path.join(args.results_dir, sys_slug)
    os.makedirs(sys_dir, exist_ok=True)

    print("==================================================")
    print(f"Active System Identifier: {sys_slug}")
    print(f"  CPU: {profile['cpu']}")
    print(f"  RAM: {profile['ram']}")
    print(f"  GPU: {profile['gpu_model']} ({profile['gpu_vram']})")
    print(f"  OS:  {profile['os']}")
    print("==================================================")

    standard_prompt = (
        "Explain the physical principles of how an airplane wing generates lift, "
        "focusing on pressure differentials and airflow speed. Be concise and thorough."
    )

    results = []
    models_to_clean = []

    for model in args.models:
        try:
            # Check if model needs to be pulled
            try:
                send_ollama_request(args.endpoint, model, "ping", num_predict=1)
            except Exception:
                pull_ollama_model(model)
                models_to_clean.append(model)

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

    # Delete models if requested
    if args.delete_after:
        for model in set(args.models + models_to_clean):
            delete_ollama_model(model)

    if results:
        # Save per-system output
        json_path = os.path.join(sys_dir, "benchmark_results.json")
        full_payload = {
            "system_profile": profile,
            "benchmark_results": results,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_payload, f, indent=2)

        md_content = format_system_markdown_table(profile, results)
        md_path = os.path.join(sys_dir, "summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"\n[✓] Saved system data to {json_path}")
        print(f"[✓] Saved system summary to {md_path}")

        # Regenerate global comparative leaderboard
        leaderboard_content = aggregate_all_systems(args.results_dir)
        leaderboard_path = os.path.join(args.results_dir, "LEADERBOARD.md")
        with open(leaderboard_path, "w", encoding="utf-8") as f:
            f.write(leaderboard_content)
        print(f"[✓] Updated comparative leaderboard at {leaderboard_path}")
        print("\n" + md_content)


if __name__ == "__main__":
    main()
