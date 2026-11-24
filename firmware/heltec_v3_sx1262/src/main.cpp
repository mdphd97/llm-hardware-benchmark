/**
 * Heltec WiFi LoRa 32 V3 (ESP32-S3 + SX1262 + SSD1306) Edge AI & LoRa Telemetry Firmware
 *
 * Dual-role build via PlatformIO environments:
 *   - heltec_v3_edge_node:      Runs INT8 quantized anomaly scoring on-chip, displays status on OLED,
 *                               transmits compact JSON over SX1262, and enters esp_deep_sleep.
 *   - heltec_v3_gateway_bridge: Receives SX1262 frames, appends hardware RSSI/SNR, and streams
 *                               newline-delimited JSON over USB-C UART to lora-agent-pipeline.
 */

#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <RadioLib.h>
#include <U8g2lib.h>
#include <esp_sleep.h>
#include <esp_timer.h>

// ---------------------------------------------------------------------------
// Heltec WiFi LoRa 32 V3 Pinout (ESP32-S3FN8 + SX1262 + SSD1306)
// ---------------------------------------------------------------------------
#define PIN_VEXT_CTRL  36  // Active LOW to power OLED & external sensors
#define PIN_OLED_SDA   17
#define PIN_OLED_SCL   18
#define PIN_OLED_RST   21

#define PIN_LORA_SCK   9
#define PIN_LORA_MISO  11
#define PIN_LORA_MOSI  10
#define PIN_LORA_NSS   8
#define PIN_LORA_RST   12
#define PIN_LORA_BUSY  13
#define PIN_LORA_DIO1  14

#define PIN_ADC_CTRL   37
#define PIN_VBAT_ADC   1

#ifndef LORA_FREQ_MHZ
#define LORA_FREQ_MHZ  915.0
#endif
#ifndef LORA_SF
#define LORA_SF        7
#endif
#ifndef LORA_BW_KHZ
#define LORA_BW_KHZ    125.0
#endif
#ifndef LORA_CR
#define LORA_CR        5
#endif

#define SLEEP_DURATION_US (15ULL * 1000000ULL)  // 15-second duty cycle for endurance tests

SPIClass loraSPI(HSPI);
SX1262 radio = new Module(PIN_LORA_NSS, PIN_LORA_DIO1, PIN_LORA_RST, PIN_LORA_BUSY, loraSPI);
U8G2_SSD1306_128X64_NONAME_F_HW_I2C oled(U8G2_R0, PIN_OLED_RST, PIN_OLED_SCL, PIN_OLED_SDA);

// ---------------------------------------------------------------------------
// INT8 Quantized TinyML Edge Classifier (4-feature linear/MLP dot-product kernel)
// Mirrors ESP-NN / TFLite Micro INT8 tensor quantization on ESP32-S3.
// Features: [temp_norm, vib_rms_norm, acoustic_norm, delta_vib_norm]
// ---------------------------------------------------------------------------
struct TinyMLResult {
    bool anomaly;
    float score;
    int64_t inference_us;
    uint32_t free_heap_bytes;
};

static const int8_t WEIGHTS_INT8[4] = {24, 58, 42, 31};
static const int32_t BIAS_INT32 = -4200;

TinyMLResult run_int8_edge_inference(float temp_c, float vib_rms, float acoustic_db, float prev_vib) {
    int64_t t_start = esp_timer_get_time();

    // Quantize sensor features into signed INT8 [-128, 127]
    int8_t x_q[4];
    x_q[0] = (int8_t)constrain((int)((temp_c - 25.0f) * 1.6f), -128, 127);
    x_q[1] = (int8_t)constrain((int)(vib_rms * 20.0f), -128, 127);
    x_q[2] = (int8_t)constrain((int)((acoustic_db - 45.0f) * 2.0f), -128, 127);
    x_q[3] = (int8_t)constrain((int)(fabsf(vib_rms - prev_vib) * 30.0f), -128, 127);

    int32_t acc = BIAS_INT32;
    for (int i = 0; i < 4; ++i) {
        acc += (int32_t)x_q[i] * (int32_t)WEIGHTS_INT8[i];
    }

    // Sigmoid approximation to [0.0, 1.0] probability score
    float logit = (float)acc / 1000.0f;
    float prob = 1.0f / (1.0f + expf(-logit));

    int64_t t_end = esp_timer_get_time();

    TinyMLResult res;
    res.anomaly = (prob >= 0.65f);
    res.score = prob;
    res.inference_us = (t_end - t_start);
    res.free_heap_bytes = esp_get_free_heap_size();
    return res;
}

uint16_t read_battery_mv() {
    pinMode(PIN_ADC_CTRL, OUTPUT);
    digitalWrite(PIN_ADC_CTRL, HIGH);
    delay(5);
    uint32_t raw_mv = analogReadMilliVolts(PIN_VBAT_ADC);
    digitalWrite(PIN_ADC_CTRL, LOW);
    // Heltec V3 voltage divider factor (~4.9x)
    return (uint16_t)(raw_mv * 4.9f);
}

void enable_vext(bool on) {
    pinMode(PIN_VEXT_CTRL, OUTPUT);
    digitalWrite(PIN_VEXT_CTRL, on ? LOW : HIGH);
}

void init_radio() {
    loraSPI.begin(PIN_LORA_SCK, PIN_LORA_MISO, PIN_LORA_MOSI, PIN_LORA_NSS);
    int state = radio.begin(LORA_FREQ_MHZ, LORA_BW_KHZ, LORA_SF, LORA_CR, RADIOLIB_SX126X_SYNC_WORD_PRIVATE, 14, 8);
    if (state != RADIOLIB_ERR_NONE) {
        Serial.printf("{\"error\":\"SX1262 init failed code %d\"}\n", state);
        while (true) { delay(1000); }
    }
    radio.setCRC(true);
}

void draw_oled_status(const char* title, const char* line1, const char* line2, const char* line3) {
    oled.clearBuffer();
    oled.setFont(u8g2_font_6x12_tf);
    oled.drawStr(0, 12, title);
    oled.drawHLine(0, 15, 128);
    oled.drawStr(0, 30, line1);
    oled.drawStr(0, 44, line2);
    oled.drawStr(0, 58, line3);
    oled.sendBuffer();
}

void setup() {
    Serial.begin(115200);
    delay(200);

    enable_vext(true);
    delay(50);

    Wire.begin(PIN_OLED_SDA, PIN_OLED_SCL);
    oled.begin();
    init_radio();

#ifdef ROLE_EDGE_NODE
    // Sample synthetic or ADC-connected bearing sensors
    float temp_c = 78.4f;
    float vib_rms = 3.45f;
    float acoustic_db = 91.2f;
    float prev_vib = 0.42f;

    TinyMLResult ml = run_int8_edge_inference(temp_c, vib_rms, acoustic_db, prev_vib);
    uint16_t vbat_mv = read_battery_mv();

    char payload[256];
    snprintf(
        payload,
        sizeof(payload),
        "{\"node_id\":\"heltec-v3-01\",\"sf\":%d,\"temp_c\":%.1f,\"vibration_rms\":%.2f,"
        "\"acoustic_db\":%.1f,\"battery_mv\":%u,\"tinyml_us\":%lld,\"score\":%.2f,"
        "\"free_heap\":%u,\"anomaly\":%s}",
        LORA_SF,
        temp_c,
        vib_rms,
        acoustic_db,
        vbat_mv,
        (long long)ml.inference_us,
        ml.score,
        ml.free_heap_bytes,
        ml.anomaly ? "true" : "false"
    );

    int tx_state = radio.transmit(payload);

    char l1[32], l2[32], l3[32];
    snprintf(l1, sizeof(l1), "ML: %s (%.2f)", ml.anomaly ? "ANOMALY" : "NOMINAL", ml.score);
    snprintf(l2, sizeof(l2), "Lat: %lld us | %uB", (long long)ml.inference_us, strlen(payload));
    snprintf(l3, sizeof(l3), "LoRa SF%d TX: %s", LORA_SF, tx_state == RADIOLIB_ERR_NONE ? "OK" : "ERR");
    draw_oled_status("Heltec V3 TinyML Edge", l1, l2, l3);
    Serial.println(payload);

    delay(1500);
    // Cut Vext OLED power and put SX1262 + ESP32-S3 into deep sleep
    radio.sleep(false);
    enable_vext(false);
    esp_sleep_enable_timer_wakeup(SLEEP_DURATION_US);
    esp_deep_sleep_start();
#else
    draw_oled_status("Heltec V3 Gateway", "SX1262 RX Ready", "SF7 BW125kHz", "Bridge -> Host LLM");
#endif
}

void loop() {
#ifdef ROLE_GATEWAY_BRIDGE
    String rx_str;
    int state = radio.receive(rx_str);
    if (state == RADIOLIB_ERR_NONE && rx_str.length() > 0) {
        float rssi = radio.getRSSI();
        float snr = radio.getSNR();

        // Inject RF link telemetry into JSON object before forwarding over USB-C UART
        if (rx_str.endsWith("}")) {
            rx_str.remove(rx_str.length() - 1);
            char rf_tail[64];
            snprintf(rf_tail, sizeof(rf_tail), ",\"rssi\":%.1f,\"snr\":%.1f}", rssi, snr);
            rx_str += rf_tail;
        }
        Serial.println(rx_str);

        char l1[32], l2[32];
        snprintf(l1, sizeof(l1), "RX: %u bytes", rx_str.length());
        snprintf(l2, sizeof(l2), "RSSI:%.0f SNR:%.1f", rssi, snr);
        draw_oled_status("Heltec V3 Gateway", l1, l2, "Forwarded -> USB");
    }
#endif
}
