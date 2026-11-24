# Heltec WiFi LoRa 32 V3 (ESP32-S3 + SX1262) Firmware

Ready-to-flash **PlatformIO** firmware pairing on-device INT8 edge anomaly scoring with SX1262 LoRa transmission and deep-sleep duty cycling, designed to feed directly into [`lora-agent-pipeline/`](../../lora-agent-pipeline/).

---

## Hardware Pinout Mapping (WiFi LoRa 32 V3)

| Subsystem | Signal | ESP32-S3 GPIO |
| :--- | :--- | :---: |
| **SX1262 LoRa** | `SCK` / `MISO` / `MOSI` / `NSS` | `9` / `11` / `10` / `8` |
| **SX1262 Control** | `RST` / `BUSY` / `DIO1` | `12` / `13` / `14` |
| **SSD1306 OLED** | `SDA` / `SCL` / `RST` | `17` / `18` / `21` |
| **Power Control** | `Vext` (Active LOW) | `36` |
| **Battery ADC** | `ADC_Ctrl` / `VBAT_ADC` | `37` / `1` |

---

## Build & Flash Environments

```bash
# 1. Flash the battery-powered Edge Inference + Deep-Sleep Sensor Node
pio run -e heltec_v3_edge_node -t upload

# 2. Flash the USB-C Gateway Receiver connected to the Linux Docker host
pio run -e heltec_v3_gateway_bridge -t upload
```

## Telemetry Schema Emitted Over LoRa & USB UART

```json
{
  "node_id": "heltec-v3-01",
  "sf": 7,
  "temp_c": 78.4,
  "vibration_rms": 3.45,
  "acoustic_db": 91.2,
  "battery_mv": 3980,
  "tinyml_us": 14,
  "score": 0.91,
  "free_heap": 294812,
  "anomaly": true,
  "rssi": -74.0,
  "snr": 9.5
}
```
