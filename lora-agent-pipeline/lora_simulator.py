#!/usr/bin/env python3
"""
Semtech SX1262 LoRa Telemetry & Time-on-Air (ToA) Simulator.
Models realistic packet airtime, RSSI/SNR, and physical-world sensor anomalies.
"""

import math
import random
import time
from typing import Any, Dict, List


def calculate_lora_toa_ms(
    payload_bytes: int = 24,
    sf: int = 7,
    bw_khz: int = 125,
    cr: int = 1,
    preamble_len: int = 8,
    crc: int = 1,
    explicit_header: int = 1,
) -> float:
    """
    Calculate exact Semtech LoRa Time-on-Air (ToA) in milliseconds.
    cr: 1 = 4/5, 2 = 4/6, 3 = 4/7, 4 = 4/8
    """
    t_sym = (2 ** sf) / bw_khz  # ms
    t_preamble = (preamble_len + 4.25) * t_sym

    ih = 0 if explicit_header else 1
    de = 1 if (sf >= 11 and bw_khz == 125) else 0

    num = 8 * payload_bytes - 4 * sf + 28 + 16 * crc - 20 * ih
    den = 4 * (sf - 2 * de)
    payload_sym_nb = 8 + max(math.ceil(num / den) * (cr + 4), 0)
    t_payload = payload_sym_nb * t_sym
    return round(t_preamble + t_payload, 2)


def generate_sensor_packet(
    node_id: str, sf: int = 7, anomaly: bool = False
) -> Dict[str, Any]:
    """Generate a structured industrial/environmental LoRa sensor telemetry packet."""
    payload_bytes = 24
    toa_ms = calculate_lora_toa_ms(payload_bytes=payload_bytes, sf=sf, bw_khz=125, cr=1)

    if anomaly:
        temp_c = round(random.uniform(74.0, 96.5), 1)
        vibration_rms = round(random.uniform(2.8, 6.4), 2)
        acoustic_db = round(random.uniform(86.0, 102.0), 1)
        status = "CRITICAL_BEARING_ANOMALY"
    else:
        temp_c = round(random.uniform(28.0, 42.0), 1)
        vibration_rms = round(random.uniform(0.15, 0.55), 2)
        acoustic_db = round(random.uniform(48.0, 62.0), 1)
        status = "NOMINAL"

    rssi = random.randint(-112, -72)
    snr = round(random.uniform(-6.5, 10.5), 1)
    battery_mv = random.randint(3350, 4120)

    telemetry_text = (
        f"Node {node_id} status={status} temp={temp_c}C "
        f"vibration={vibration_rms}g acoustic={acoustic_db}dB "
        f"rssi={rssi}dBm snr={snr}dB battery={battery_mv}mV"
    )

    return {
        "timestamp": time.time(),
        "node_id": node_id,
        "sf": sf,
        "payload_bytes": payload_bytes,
        "toa_ms": toa_ms,
        "temp_c": temp_c,
        "vibration_rms": vibration_rms,
        "acoustic_db": acoustic_db,
        "rssi": rssi,
        "snr": snr,
        "battery_mv": battery_mv,
        "anomaly": anomaly,
        "telemetry_text": telemetry_text,
    }


def generate_fleet_burst(
    num_nodes: int = 20, anomaly_ratio: float = 0.2, sf: int = 7
) -> List[Dict[str, Any]]:
    """Generate a burst of telemetry packets from a distributed LoRa sensor fleet."""
    packets = []
    for i in range(1, num_nodes + 1):
        node_id = f"lora-node-{i:02d}"
        is_anomaly = random.random() < anomaly_ratio
        packets.append(generate_sensor_packet(node_id=node_id, sf=sf, anomaly=is_anomaly))
    return packets
