#!/usr/bin/env python3
"""
LoRa-to-Local-LLM Agent Pipeline Stress Test.
Bridges simulated LoRa telemetry -> Ollama Embeddings -> Qdrant Vector Retrieval -> MCP Tool-Calling Loop.
Measures the complete Sensor-to-Agent Latency Waterfall and Host Energy Profile.
"""

import argparse
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List

from energy_profiler import EnergyMonitor
from lora_simulator import calculate_lora_toa_ms, generate_fleet_burst


COLLECTION_NAME = "lora_telemetry"
EMBED_MODEL = "nomic-embed-text:latest"


def parse_gateway_json_line(raw_line: str) -> Dict[str, Any]:
    """
    Parse newline-delimited JSON emitted by Heltec WiFi LoRa 32 V3 USB-C Gateway
    and normalize into the pipeline telemetry schema.
    """
    raw = json.loads(raw_line.strip())
    node_id = str(raw.get("node_id", "heltec-v3-unknown"))
    sf = int(raw.get("sf", 7))
    temp_c = float(raw.get("temp_c", 0.0))
    vib = float(raw.get("vibration_rms", 0.0))
    acoustic = float(raw.get("acoustic_db", 0.0))
    rssi = float(raw.get("rssi", -80.0))
    snr = float(raw.get("snr", 5.0))
    vbat = int(raw.get("battery_mv", 3800))
    anomaly = bool(raw.get("anomaly", False))
    status = "CRITICAL_BEARING_ANOMALY" if anomaly else "NOMINAL"

    raw["toa_ms"] = raw.get("toa_ms") or calculate_lora_toa_ms(len(raw_line), sf=sf)
    raw["telemetry_text"] = raw.get("telemetry_text") or (
        f"Node {node_id} status={status} temp={temp_c}C "
        f"vibration={vib}g acoustic={acoustic}dB "
        f"rssi={rssi}dBm snr={snr}dB battery={vbat}mV"
    )
    return raw


def read_serial_packets(port: str, baud: int, max_packets: int) -> List[Dict[str, Any]]:
    """Read newline-delimited JSON packets from a Linux USB serial device without external deps."""
    import subprocess as sp

    sp.check_call(["stty", "-F", port, str(baud), "raw", "-echo"])
    packets: List[Dict[str, Any]] = []
    print(f"[*] Listening for {max_packets} live Heltec packets on {port} @ {baud} baud...")
    with open(port, "r", encoding="utf-8", errors="replace") as ser:
        while len(packets) < max_packets:
            line = ser.readline()
            if not line or not line.strip().startswith("{"):
                continue
            try:
                packets.append(parse_gateway_json_line(line))
            except Exception as exc:
                print(f"[!] Skipping malformed serial frame: {exc}")
    return packets


def http_json(url: str, method: str = "GET", payload: Any = None, timeout: int = 120) -> Dict[str, Any]:
    """Zero-dependency HTTP JSON helper."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ensure_qdrant_collection(qdrant_url: str, vector_size: int = 768) -> None:
    """Create or recreate Qdrant collection for clean benchmark runs."""
    base = qdrant_url.rstrip("/")
    try:
        http_json(f"{base}/collections/{COLLECTION_NAME}", method="DELETE")
    except Exception:
        pass
    http_json(
        f"{base}/collections/{COLLECTION_NAME}",
        method="PUT",
        payload={"vectors": {"size": vector_size, "distance": "Cosine"}},
    )


def get_embedding(ollama_url: str, text: str) -> List[float]:
    """Generate vector embedding via local Ollama nomic-embed-text."""
    res = http_json(
        f"{ollama_url.rstrip('/')}/api/embeddings",
        method="POST",
        payload={"model": EMBED_MODEL, "prompt": text},
    )
    return res["embedding"]


def upsert_qdrant_point(qdrant_url: str, point_id: int, vector: List[float], payload: Dict[str, Any]) -> None:
    """Insert telemetry vector and metadata into Qdrant."""
    http_json(
        f"{qdrant_url.rstrip('/')}/collections/{COLLECTION_NAME}/points?wait=true",
        method="PUT",
        payload={"points": [{"id": point_id, "vector": vector, "payload": payload}]},
    )


def search_qdrant(qdrant_url: str, vector: List[float], limit: int = 3) -> List[Dict[str, Any]]:
    """Query Qdrant for similar historical telemetry events."""
    res = http_json(
        f"{qdrant_url.rstrip('/')}/collections/{COLLECTION_NAME}/points/search",
        method="POST",
        payload={"vector": vector, "limit": limit, "with_payload": True},
    )
    return res.get("result", [])


def dispatch_mcp_reasoning(ollama_url: str, llm_model: str, alert_text: str, context_hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run local LLM reasoning with an MCP-style structured tool-call schema
    to decide physical-world actuation (e.g., shutdown_motor, log_maintenance).
    """
    context_summary = "\n".join(
        f"- {hit['payload']['telemetry_text']}" for hit in context_hits
    )
    prompt = (
        "You are an industrial edge agent monitoring LoRa sensor nodes.\n"
        "Available MCP tools: [trigger_emergency_shutdown(node_id), schedule_inspection(node_id), acknowledge_nominal(node_id)].\n"
        f"Incoming Alert:\n{alert_text}\n\n"
        f"Historical Vector Context:\n{context_summary}\n\n"
        "Respond strictly with a single JSON object containing 'tool' and 'node_id' and 'reason'."
    )
    res = http_json(
        f"{ollama_url.rstrip('/')}/api/generate",
        method="POST",
        payload={
            "model": llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 96, "temperature": 0.0},
        },
    )
    return res


def run_stress_test(
    ollama_url: str,
    qdrant_url: str,
    llm_model: str,
    num_nodes: int,
    sf: int,
    delete_after: bool,
    serial_port: str = "",
    baud: int = 115200,
) -> Dict[str, Any]:
    """Execute full LoRa-to-Local-LLM Agent Pipeline benchmark."""
    # Ensure LLM is available if needed
    pulled_temp = False
    try:
        http_json(
            f"{ollama_url.rstrip('/')}/api/generate",
            method="POST",
            payload={"model": llm_model, "prompt": "ping", "stream": False, "options": {"num_predict": 1}},
        )
    except Exception:
        print(f"[*] Pulling '{llm_model}' for MCP agent reasoning...")
        subprocess.check_call(["ollama", "pull", llm_model])
        pulled_temp = True

    print(f"[*] Initializing Qdrant collection '{COLLECTION_NAME}'...")
    sample_vec = get_embedding(ollama_url, "init")
    ensure_qdrant_collection(qdrant_url, vector_size=len(sample_vec))

    if serial_port:
        packets = read_serial_packets(serial_port, baud, max_packets=num_nodes)
    else:
        print(f"[*] Generating burst of {num_nodes} LoRa packets (SX1262, SF{sf})...")
        packets = generate_fleet_burst(num_nodes=num_nodes, anomaly_ratio=0.25, sf=sf)

    monitor = EnergyMonitor()
    monitor.start()

    embed_latencies_ms = []
    upsert_latencies_ms = []
    search_latencies_ms = []
    llm_latencies_ms = []
    toa_latencies_ms = []
    anomalies_processed = 0

    for idx, pkt in enumerate(packets, start=1):
        toa_latencies_ms.append(pkt["toa_ms"])

        # 1. Embedding
        t0 = time.perf_counter()
        vec = get_embedding(ollama_url, pkt["telemetry_text"])
        t1 = time.perf_counter()
        embed_latencies_ms.append((t1 - t0) * 1000.0)

        # 2. Vector Upsert
        t2 = time.perf_counter()
        upsert_qdrant_point(qdrant_url, idx, vec, pkt)
        t3 = time.perf_counter()
        upsert_latencies_ms.append((t3 - t2) * 1000.0)

        # 3. If anomaly, run Vector Retrieval + MCP LLM Reasoning Loop
        if pkt["anomaly"]:
            anomalies_processed += 1
            t4 = time.perf_counter()
            hits = search_qdrant(qdrant_url, vec, limit=3)
            t5 = time.perf_counter()
            search_latencies_ms.append((t5 - t4) * 1000.0)

            t6 = time.perf_counter()
            dispatch_mcp_reasoning(ollama_url, llm_model, pkt["telemetry_text"], hits)
            t7 = time.perf_counter()
            llm_latencies_ms.append((t7 - t6) * 1000.0)

        if idx % 5 == 0:
            monitor.sample()

    energy_stats = monitor.stop()

    if delete_after and pulled_temp:
        print(f"[*] Cleaning up temporary model '{llm_model}'...")
        subprocess.call(["ollama", "rm", llm_model], stdout=subprocess.DEVNULL)

    def avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else 0.0

    summary = {
        "config": {
            "num_nodes": num_nodes,
            "lora_sf": sf,
            "embed_model": EMBED_MODEL,
            "llm_model": llm_model,
            "vector_db": "Qdrant (Docker)",
        },
        "latency_waterfall_ms": {
            "avg_lora_toa_ms": avg(toa_latencies_ms),
            "avg_embedding_ms": avg(embed_latencies_ms),
            "avg_qdrant_upsert_ms": avg(upsert_latencies_ms),
            "avg_qdrant_search_ms": avg(search_latencies_ms),
            "avg_llm_mcp_reason_ms": avg(llm_latencies_ms),
            "total_anomaly_e2e_ms": round(
                avg(toa_latencies_ms)
                + avg(embed_latencies_ms)
                + avg(search_latencies_ms)
                + avg(llm_latencies_ms),
                2,
            ),
        },
        "throughput": {
            "packets_ingested": num_nodes,
            "anomalies_triaged": anomalies_processed,
            "ingest_rate_pkt_s": round(num_nodes / max(energy_stats["duration_s"], 1e-6), 2),
        },
        "energy_profile": energy_stats,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="LoRa-to-Local-LLM Agent Pipeline Stress Test")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--llm-model", default="llama3.2:1b")
    parser.add_argument("--nodes", type=int, default=20, help="Number of simulated LoRa nodes or live packets")
    parser.add_argument("--sf", type=int, default=7, help="LoRa Spreading Factor (7-12)")
    parser.add_argument("--serial-port", default="", help="Optional USB serial device (e.g. /dev/ttyUSB0) for live Heltec V3 gateway")
    parser.add_argument("--baud", type=int, default=115200, help="USB serial baud rate (default: 115200)")
    parser.add_argument("--delete-after", action="store_true", help="Remove pulled LLM weights after test")
    parser.add_argument("--output-json", default="../results/lora_agent_stress_test.json")
    args = parser.parse_args()

    results = run_stress_test(
        ollama_url=args.ollama_url,
        qdrant_url=args.qdrant_url,
        llm_model=args.llm_model,
        num_nodes=args.nodes,
        sf=args.sf,
        delete_after=args.delete_after,
        serial_port=args.serial_port,
        baud=args.baud,
    )

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    wf = results["latency_waterfall_ms"]
    en = results["energy_profile"]
    print("\n==================================================")
    print("LoRa-to-Local-LLM Pipeline Stress Test Results")
    print("==================================================")
    print(f"Packets Ingested:     {results['throughput']['packets_ingested']} (Anomalies: {results['throughput']['anomalies_triaged']})")
    print(f"LoRa Time-on-Air:     {wf['avg_lora_toa_ms']} ms")
    print(f"Ollama Embedding:     {wf['avg_embedding_ms']} ms")
    print(f"Qdrant Upsert/Search: {wf['avg_qdrant_upsert_ms']} ms / {wf['avg_qdrant_search_ms']} ms")
    print(f"LLM MCP Reasoning:    {wf['avg_llm_mcp_reason_ms']} ms")
    print(f"Total E2E Anomaly:    {wf['total_anomaly_e2e_ms']} ms")
    print(f"GPU Power / Energy:   {en['avg_gpu_watts']} W avg | {en['gpu_energy_joules']} J total | Peak {en['peak_gpu_temp_c']} C")
    print("==================================================")


if __name__ == "__main__":
    main()
