/*
  FC.ino — Teensy 4.1 Production Flight Computer Firmware
  =============================================================================
  Hardware Stack:
    - MCU: Teensy 4.1 (600 MHz Cortex-M7)
    - Primary RF: Digi XBee 3 Pro (XB3-24AST) in API 1 Mode on Serial2
    - Long-Range RF: Reyax RYLR998 (SX1262 22 dBm) on Serial8
    - GNSS: 7Semi L89HA on Serial7 (9600 baud)
    - IMU: ST LSM6DSO32 (±32g Accel, ±2000 dps Gyro) via I2C0
    - Barometer: Bosch BMP581 (100 Hz Continuous, 8x OSR) via I2C0
    - Actuators: Dual D4184 MOSFETs (Main Solenoid Pin 23, Drogue Solenoid Pin 22)
    - Storage: Onboard MicroSD via Native 4-bit SDIO (BUILTIN_SDCARD)

  UPLINK COMMANDS (single ASCII character via XBee RF data, or "PING"/"$PING"):
    'A' — Start XBee telemetry downlink (absolute, safe to repeat)
    'B' — Stop  XBee telemetry downlink (refused between launch and landing)
    'E' — Manual drogue eject override (fires pin 22 unconditionally;
          accepted in any FSM state, even if drogue already fired)
    PING / $PING — Link test; rocket replies with $TEAM_ID,PONG,count,state,V*XX

  Commands are accepted on XBee receive frames 0x80 (64-bit), 0x81 (16-bit) and
  0x90 (extended). 'B' is refused mid-flight so a stray key can never silence the
  vehicle. Telemetry is always ON after any reset (watchdog or power glitch).
  The LoRa recovery beacon never stops regardless of 'A'/'B' state.
  =============================================================================
*/

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <EEPROM.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#if __has_include(<Watchdog_t4.h>)
  #include <Watchdog_t4.h>
  #define HAVE_WDT 1
  WDT_T4<WDT1> wdt;
#else
  #define HAVE_WDT 0
#endif

// ============================================================================
// 1. MASTER CONFIGURATION: RULES, THRESHOLDS, FLAGS & PARAMETERS
// ============================================================================

// --- Identity & Ground Station ---
static const char TEAM_ID[]             = "036";

// --- Feature Enable / Disable Flags ---
constexpr bool ENABLE_WATCHDOG            = true;
constexpr bool ENABLE_SD_LOGGING          = true;
constexpr bool ENABLE_EEPROM_RESUME       = true;
constexpr bool REQUIRE_DROGUE_FOR_MAIN    = true;   // Main only deploys if drogue already fired
constexpr bool ENABLE_GCS_PING_BUZZ       = true;   // Chirp buzzer on GCS PING
constexpr bool REPORT_FSM_FLAGS           = true;   // SOLENOID/NICHROME follow drogueFired/mainFired
constexpr bool ALLOW_STOP_IN_FLIGHT       = false;  // 'B' refused between launch and landing

// --- Digi XBee 3 Pro (XB3-24AST) 802.15.4 Ground Station MAC Addresses ---
// Flight Computer Self MAC: 00 13 A2 00 42 8D CA 31  (this Teensy, TX)
// CONFIRM these 64-bit addresses against ATSH/ATSL printed on each ground XBee.
const uint8_t XBEE_MAC_RX1[8]             = {0x00, 0x13, 0xA2, 0x00, 0x42, 0xA0, 0x95, 0xF5}; // RX1 (0013A20042A095F5)
const uint8_t XBEE_MAC_RX2[8]             = {0x00, 0x13, 0xA2, 0x00, 0x42, 0xA0, 0x96, 0x41}; // RX2 (0013A20042A09641)

// --- Reyax RYLR998 LoRa Radio Plan (SX1262 Transceiver) ---
constexpr uint32_t LORA_BAND_HZ           = 867500000UL; // 867.500 MHz (India 865-867 MHz band)
constexpr uint8_t  LORA_SF                = 10;
constexpr uint8_t  LORA_BW_CODE           = 7;           // 7 = 125 kHz
constexpr uint8_t  LORA_CR_CODE           = 1;           // 1 = 4/5
constexpr uint8_t  LORA_PREAMBLE          = 12;
constexpr uint8_t  LORA_NETWORKID         = 6;
constexpr uint8_t  LORA_ADDRESS           = 1;
constexpr uint8_t  LORA_POWER_DBM         = 22;          // RYLR998 max 22 dBm

// --- Cadence & Timing (ms) ---
constexpr uint32_t SENSOR_PERIOD_MS       = 10;     // 100 Hz
constexpr uint32_t TX_PERIOD_MS           = 50;     // 20 Hz XBee downlink
constexpr uint32_t LORA_PERIOD_MS         = 1000;   // 1 Hz LoRa beacon
constexpr uint32_t STATUS_PERIOD_MS       = 1000;   // 1 Hz USB diagnostics
constexpr uint32_t SD_FLUSH_PERIOD_MS     = 100;    // 10 Hz SD flush
constexpr uint32_t BATT_SAMPLE_MS         = 100;    // 10 Hz ADC

// --- Hardware Pin Allocations ---
#define GNSS_SERIAL Serial7   // RX7: Pin 28, TX7: Pin 29
#define LORA_SERIAL Serial8   // RX8: Pin 34, TX8: Pin 35
#define XBEE_SERIAL Serial2   // RX2: Pin 7,  TX2: Pin 8
constexpr uint32_t GNSS_BAUD  = 9600;
constexpr uint32_t LORA_BAUD  = 115200;
constexpr uint32_t XBEE_BAUD  = 115200;

constexpr uint8_t PIN_BUZZER              = 4;
constexpr uint8_t PIN_ADC_FC_2S           = 24;     // A10: 2S Flight Computer Battery
constexpr uint8_t PIN_ADC_SOL_3S          = 25;     // A11: 3S Solenoid Battery
constexpr uint8_t PIN_LED_R               = 30;
constexpr uint8_t PIN_LED_G               = 31;
constexpr uint8_t PIN_LED_B               = 32;
constexpr uint8_t PIN_I2C_SDA             = 18;
constexpr uint8_t PIN_I2C_SCL             = 19;
constexpr uint8_t PIN_SOL_MAIN            = 23;     // D4184 #1 Main Parachute Solenoid
constexpr uint8_t PIN_SOL_DROGUE          = 22;     // D4184 #2 Drogue Parachute Solenoid (eject pin)
constexpr uint8_t SD_CS_PIN               = BUILTIN_SDCARD;

// --- Actuator Timers ---
constexpr uint32_t MAIN_FIRE_DURATION_MS  = 3000;   // 3 s main solenoid pulse
constexpr uint32_t DROGUE_MAX_HOLD_MS     = 45000;  // 45 s ascent failsafe cutoff

// --- Flight State Machine Thresholds ---
constexpr float    GRAVITY_EARTH          = 9.80665f;
constexpr float    LAUNCH_ACC_MS2         = 2.5f * GRAVITY_EARTH;  // 2.5 g
constexpr uint8_t  LAUNCH_SAMPLES         = 5;       // 50 ms sustained
constexpr float    LAUNCH_CONFIRM_M       = 10.0f;   // 10 m altitude gain
constexpr uint32_t LAUNCH_CONFIRM_MS      = 2500;    // 2.5 s window
constexpr uint32_t MOTOR_BURNOUT_MS       = 3000;    // 3 s boost lockout (no apogee check)
constexpr float    APOGEE_DROP_M          = 3.0f;    // 3 m drop from peak
constexpr uint8_t  APOGEE_SAMPLES         = 6;       // 6 consecutive descent samples
constexpr uint32_t DEPLOY_HOLD_MS         = 1000;    // hold in DEPLOY before DESCENT
constexpr float    MAIN_DEPLOY_AGL_M      = 400.0f;  // main deploy altitude AGL — verify vs CDR
constexpr float    LAND_VS_MS             = 0.7f;    // stillness speed limit (m/s)
constexpr uint8_t  LAND_STILL_WINDOWS     = 3;       // 3 consecutive 1-s stillness windows
constexpr float    LAND_MAX_AGL_M         = 60.0f;   // must be below 60 m AGL to declare landing
constexpr float    RESUME_MIN_AGL_M       = 30.0f;
constexpr float    RESUME_MAX_AGL_M       = 6000.0f;

// --- Pad Calibration ---
constexpr uint16_t CAL_SAMPLES            = 300;     // 3 s
constexpr float    CAL_GYRO_MAX_DPS       = 15.0f;
constexpr float    CAL_ACC_MIN_MS2        = 0.7f * GRAVITY_EARTH;
constexpr float    CAL_ACC_MAX_MS2        = 1.3f * GRAVITY_EARTH;

// --- Sensor Plausibility Windows ---
constexpr float    PA_MIN                 = 30000.0f;
constexpr float    PA_MAX                 = 110000.0f;
constexpr float    TC_MIN                 = -60.0f;
constexpr float    TC_MAX                 = 100.0f;
constexpr float    ACC_ABS_MAX_MS2        = 33.0f * GRAVITY_EARTH;
constexpr float    GYRO_ABS_MAX_DPS       = 2100.0f;

// --- Sensor Addresses & Scales ---
constexpr uint8_t  LSM_ADDR               = 0x6A;
constexpr uint8_t  LSM_WHO_AM_I           = 0x0F;
constexpr uint8_t  LSM_CTRL1_XL           = 0x10;
constexpr uint8_t  LSM_CTRL2_G            = 0x11;
constexpr uint8_t  LSM_CTRL3_C            = 0x12;
constexpr uint8_t  LSM_OUTX_L_G           = 0x22;
constexpr uint8_t  BMP_ADDR               = 0x47;
constexpr uint8_t  BMP_CHIP_ID            = 0x01;
constexpr uint8_t  BMP_TEMP_DATA0         = 0x1D;
constexpr uint8_t  BMP_OSR_CONFIG         = 0x36;
constexpr uint8_t  BMP_ODR_CONFIG         = 0x37;
constexpr float    ACCEL_SCALE_MPS2       = (32.0f / 32768.0f) * GRAVITY_EARTH;
constexpr float    GYRO_SCALE_DPS         = 0.070f;
constexpr float    BATT_CAL_FACTOR        = (3.30f / 4095.0f) * 4.30f;  // 33k/10k divider
constexpr float    STD_SEA_LEVEL_PA       = 101325.0f;
constexpr float    BARO_EXPONENT          = 0.190294957f;

// --- GCS Flight States ---
constexpr uint8_t FSM_BOOT                = 0;
constexpr uint8_t FSM_TEST                = 1;  // reserved, not currently entered
constexpr uint8_t FSM_PAD                 = 2;
constexpr uint8_t FSM_ASCENT              = 3;
constexpr uint8_t FSM_DEPLOY              = 4;
constexpr uint8_t FSM_DESCENT             = 5;
constexpr uint8_t FSM_MAIN                = 6;  // GCS name: AEROBRAKE_RELEASE
constexpr uint8_t FSM_IMPACT              = 7;

// ============================================================================
// 2. INTERNAL BUFFERS & GLOBAL STATE
// ============================================================================
static uint8_t xbeeTxMem[1024];
static uint8_t xbeeRxMem[512];
static uint8_t loraTxMem[512];
static uint8_t loraRxMem[256];
static uint8_t gnssRxMem[1024];

// SD card — ring buffer large enough to absorb 100 Hz writes between 10 Hz flushes
#define SD_BUF_SIZE 8192
static char    sdBuffer[SD_BUF_SIZE];
static int     sdHead = 0, sdTail = 0;
static File    logFile;
static bool    sdActive = false;

// Sensor Readings
static float gyroBias[3]       = {0, 0, 0};
static float groundPa          = STD_SEA_LEVEL_PA;
static float accel[3]          = {0, 0, GRAVITY_EARTH};
static float gyro[3]           = {0, 0, 0};
static float baroPa            = STD_SEA_LEVEL_PA;
static float baroTempC         = 20.0f;
static float aglRaw = 0, aglFilt = 0, aglPrev1s = 0, vertSpeed = 0, maxAgl = 0;
static float vbatFC = 0.0f, vbatSol = 0.0f;

static bool     baroEverGood   = false, imuEverGood = false;
static uint32_t imuLastGoodMs  = 0, baroLastGoodMs = 0;
static uint8_t  i2cFailStreak  = 0;
static uint32_t i2cRecoveries  = 0, lastI2cRecoverMs = 0;
static uint32_t imuFailures    = 0, baroFailures = 0;
static bool     imuIdOk        = false, baroIdOk = false;

// GNSS
static double   gLat = 0, gLon = 0;
static float    gAlt = 0;
static uint8_t  gSats = 0, gQuality = 0;
static char     gTime[16] = "";
static uint32_t gLastGgaMs = 0;
static uint32_t nmeaGood = 0, nmeaBad = 0;

// Flight State Machine & Solenoids
static uint8_t  fsmState       = FSM_BOOT;
static bool     drogueFired    = false, mainFired = false;
static uint32_t mainFireOffMs  = 0;
static uint32_t droguePulseOffMs = 0;  // non-zero while GCS-commanded pulse is HIGH
static uint32_t tLaunchMs      = 0, tDeployMs = 0;
static bool     launchConfirmed= false;
static uint8_t  launchCnt = 0, apogeeCnt = 0, stillCnt = 0;

// Pad Calibration
static uint8_t  calPhase       = 0; // 0 resume-check, 1 collect, 2 done
static uint16_t calCount       = 0;
static float    calGyroSum[3]  = {0, 0, 0};
static double   calPaSum       = 0;
static uint32_t calPhase0Start = 0;
static bool     resumeCandidate= false;

// Telemetry & Radio Diagnostics
static uint32_t packetCount    = 0;
static uint32_t txDropped      = 0, txSentX = 0, loraSent = 0, loraDropped = 0;
static uint32_t statusOk[2]    = {0, 0}, statusFail[2] = {0, 0};
static uint8_t  lastStatusCode = 0;
static uint8_t  frameIdDest[256];
static uint8_t  nextFrameId    = 1;
static uint32_t lastStatusMs   = 0, sentSinceStatus = 0;
static uint32_t xbeeRecoveries = 0, lastXbeeRecoverMs = 0;
static uint32_t xbeeRxFrames   = 0, xbeeRxBadCks = 0, pingsAnswered = 0;

// Uplink Command State — telemetry always ON after any reset so a watchdog
// reboot in flight never leaves the vehicle silent.
static bool     txEnabled      = true;
static uint32_t cmdAccepted    = 0, cmdRejected = 0, cmdIgnored = 0;
static uint32_t lastCmdMs      = 0;
static char     lastCmd        = '-';

// Reyax RYLR998 Config State Machine
static uint8_t  loraCfgStep    = 0;
static bool     loraReady      = false;
static uint32_t loraCfgSentMs  = 0;
static bool     loraCfgWaiting = false;
static uint8_t  loraCfgErrors  = 0;

// Schedulers
static uint32_t nextSensorMs   = 0, nextTxMs = 0, nextLoraMs = 0, nextStatusMs = 0;
static uint32_t nextBattMs     = 0, nextSdFlushMs = 0, tick1sCount = 0;
static uint32_t beepUntilMs    = 0;

// ============================================================================
// 3. UTILITIES
// ============================================================================
static uint32_t crc32Bytes(const uint8_t* p, size_t n) {
  uint32_t c = 0xFFFFFFFFUL;
  for (size_t i = 0; i < n; i++) {
    c ^= p[i];
    for (uint8_t b = 0; b < 8; b++) c = (c >> 1) ^ (0xEDB88320UL & (0 - (c & 1)));
  }
  return ~c;
}

static float pressureToAlt(float pa, float refPa) {
  return 44330.0f * (1.0f - powf(pa / refPa, BARO_EXPONENT));
}

static bool finiteIn(float v, float lo, float hi) {
  return isfinite(v) && v >= lo && v <= hi;
}

// XOR of every byte between '$' and '*' — matches GCS verification
static uint8_t xorChecksum(const char* body) {
  uint8_t c = 0;
  while (*body) c ^= (uint8_t)*body++;
  return c;
}

static void beepOnce(uint16_t ms) {
  beepUntilMs = millis() + ms;
}

static void feedWatchdog() {
#if HAVE_WDT
  if (ENABLE_WATCHDOG) wdt.feed();
#endif
}

// True while any flight phase where stopping telemetry is unsafe
static bool inFlightNow() {
  return fsmState >= FSM_ASCENT && fsmState <= FSM_MAIN;
}

// ============================================================================
// 4. NON-BLOCKING SD CARD LOGGING
// ============================================================================
static void logToSdQueue(const char* str) {
  if (!ENABLE_SD_LOGGING || !sdActive) return;
  size_t len = strlen(str);
  for (size_t i = 0; i < len; i++) {
    int nextHead = (sdHead + 1) % SD_BUF_SIZE;
    if (nextHead != sdTail) {
      sdBuffer[sdHead] = str[i];
      sdHead = nextHead;
    }
    // If buffer full the oldest byte is silently dropped (production rate > drain
    // rate would be a config issue, not handled at this layer)
  }
}

// Drains the ring buffer to SD in one bulk write per contiguous region.
// Called at 10 Hz; at 100 Hz / ~128 bytes per row the buffer holds ~640 ms.
static void flushSdQueue() {
  if (!ENABLE_SD_LOGGING || !sdActive || !logFile) return;
  int head = sdHead;
  if (sdTail == head) return;
  if (head > sdTail) {
    logFile.write((const uint8_t*)&sdBuffer[sdTail], head - sdTail);
  } else {
    // Wrap-around: write tail→end, then start→head
    logFile.write((const uint8_t*)&sdBuffer[sdTail], SD_BUF_SIZE - sdTail);
    if (head > 0) logFile.write((const uint8_t*)&sdBuffer[0], head);
  }
  sdTail = head;
  logFile.flush();
}

static void initSdCard() {
  if (!ENABLE_SD_LOGGING) return;
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("[SD] Card init failed or not inserted");
    sdActive = false;
    return;
  }

  uint16_t fltCount = 0;
  EEPROM.get(100, fltCount);
  fltCount++;
  EEPROM.put(100, fltCount);

  char fn[32];
  snprintf(fn, sizeof(fn), "FLIGHT_%03u.CSV", fltCount);
  logFile = SD.open(fn, FILE_WRITE);
  if (logFile) {
    logFile.println("TIMESTAMP_MS,PKT_COUNT,STATE,AGL_M,PRESS_HPA,TEMP_C,VBAT_2S,VBAT_3S,AX,AY,AZ,GX,GY,GZ,LAT,LON,SATS,DROGUE,MAIN");
    logFile.flush();
    sdActive = true;
    Serial.printf("[SD] Log: %s\n", fn);
  } else {
    sdActive = false;
    Serial.println("[SD] Failed to open log file");
  }
}

// ============================================================================
// 5. EEPROM MID-FLIGHT RESUME
// ============================================================================
struct CalRecord {
  uint32_t magic;
  float    groundPa;
  float    gyroBias[3];
  uint8_t  inFlight;
  uint8_t  drogueFired;  // persisted so a reset after EJECT doesn't re-energise the solenoid
  uint8_t  pad[2];
  uint32_t crc;
};
constexpr uint32_t CAL_MAGIC = 0x52434B54UL; // "RCKT"
static CalRecord savedRec;

static void recordSave(uint8_t inFlight) {
  if (!ENABLE_EEPROM_RESUME) return;
  CalRecord r;
  memset(&r, 0, sizeof(r));
  r.magic       = CAL_MAGIC;
  r.groundPa    = groundPa;
  for (uint8_t i = 0; i < 3; i++) r.gyroBias[i] = gyroBias[i];
  r.inFlight    = inFlight;
  r.drogueFired = drogueFired ? 1u : 0u;
  r.crc = crc32Bytes((const uint8_t*)&r, sizeof(r) - sizeof(r.crc));
  EEPROM.put(0, r);
  savedRec = r;
}

static bool recordLoad() {
  if (!ENABLE_EEPROM_RESUME) return false;
  CalRecord r;
  EEPROM.get(0, r);
  if (r.magic != CAL_MAGIC) return false;
  if (r.crc != crc32Bytes((const uint8_t*)&r, sizeof(r) - sizeof(r.crc))) return false;
  if (!finiteIn(r.groundPa, PA_MIN, PA_MAX)) return false;
  for (uint8_t i = 0; i < 3; i++) if (!finiteIn(r.gyroBias[i], -100.0f, 100.0f)) return false;
  savedRec = r;
  return true;
}

// ============================================================================
// 6. I2C LAYER WITH AUTO-UNSTICK
// ============================================================================
static bool i2cWrite(uint8_t dev, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(dev);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool i2cRead(uint8_t dev, uint8_t reg, uint8_t* buf, uint8_t n) {
  Wire.beginTransmission(dev);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(dev, n) != n) return false;
  for (uint8_t i = 0; i < n; i++) {
    if (!Wire.available()) return false;
    buf[i] = (uint8_t)Wire.read();
  }
  return true;
}

// Verify WHO_AM_I AND all mode register writes; a partial init is treated as
// sensor absent so the I2C recovery loop will retry.
static void initSensors() {
  uint8_t id = 0;
  imuIdOk  = i2cRead(LSM_ADDR, LSM_WHO_AM_I, &id, 1) && id == 0x6C
              && i2cWrite(LSM_ADDR, LSM_CTRL3_C,  0x44)  // BDU on, addr auto-inc
              && i2cWrite(LSM_ADDR, LSM_CTRL1_XL, 0x64)  // 416 Hz, ±32 g
              && i2cWrite(LSM_ADDR, LSM_CTRL2_G,  0x6C); // 416 Hz, ±2000 dps
  baroIdOk = i2cRead(BMP_ADDR, BMP_CHIP_ID,   &id, 1) && id == 0x50
              && i2cWrite(BMP_ADDR, BMP_OSR_CONFIG, 0x58) // 8x pressure OSR
              && i2cWrite(BMP_ADDR, BMP_ODR_CONFIG, 0x2B);// 100 Hz continuous
}

// Clock 9 SCL pulses to release a slave holding SDA low, then restart Wire.
static void i2cUnstick() {
  Wire.end();
  pinMode(PIN_I2C_SDA, INPUT_PULLUP);
  pinMode(PIN_I2C_SCL, OUTPUT);
  digitalWrite(PIN_I2C_SCL, HIGH);
  for (uint8_t i = 0; i < 9 && digitalRead(PIN_I2C_SDA) == LOW; i++) {
    digitalWrite(PIN_I2C_SCL, LOW);  delayMicroseconds(5);
    digitalWrite(PIN_I2C_SCL, HIGH); delayMicroseconds(5);
  }
  pinMode(PIN_I2C_SDA, OUTPUT);
  digitalWrite(PIN_I2C_SDA, LOW);  delayMicroseconds(5);
  digitalWrite(PIN_I2C_SCL, HIGH); delayMicroseconds(5);
  digitalWrite(PIN_I2C_SDA, HIGH); delayMicroseconds(5);
  Wire.begin();
  Wire.setClock(400000);
}

static void i2cRecover() {
  uint32_t now = millis();
  if (now - lastI2cRecoverMs < 500) return;
  lastI2cRecoverMs = now;
  i2cRecoveries++;
  i2cUnstick();
  initSensors();
  i2cFailStreak = 0;
}

// ============================================================================
// 7. SENSOR ACQUISITION
// ============================================================================
static bool readImu() {
  uint8_t b[12];
  if (!i2cRead(LSM_ADDR, LSM_OUTX_L_G, b, 12)) return false;
  float g[3], a[3];
  for (uint8_t i = 0; i < 3; i++) {
    int16_t rg = (int16_t)((uint16_t)b[2 * i]     | ((uint16_t)b[2 * i + 1]     << 8));
    int16_t ra = (int16_t)((uint16_t)b[6 + 2 * i] | ((uint16_t)b[6 + 2 * i + 1] << 8));
    g[i] = (float)rg * GYRO_SCALE_DPS;
    a[i] = (float)ra * ACCEL_SCALE_MPS2;
  }
  for (uint8_t i = 0; i < 3; i++) {
    if (!finiteIn(a[i], -ACC_ABS_MAX_MS2,   ACC_ABS_MAX_MS2))   return false;
    if (!finiteIn(g[i], -GYRO_ABS_MAX_DPS,  GYRO_ABS_MAX_DPS))  return false;
  }
  for (uint8_t i = 0; i < 3; i++) {
    accel[i] = a[i];
    gyro[i]  = g[i] - gyroBias[i];
  }
  imuLastGoodMs = millis();
  imuEverGood = true;
  return true;
}

static bool readBaro() {
  uint8_t b[6];
  if (!i2cRead(BMP_ADDR, BMP_TEMP_DATA0, b, 6)) return false;
  int32_t t = (int32_t)(((uint32_t)b[2] << 16) | ((uint32_t)b[1] << 8) | b[0]);
  if (t & 0x800000) t -= (int32_t)0x1000000;  // 24-bit sign extend (defined behaviour)
  uint32_t p = ((uint32_t)b[5] << 16) | ((uint32_t)b[4] << 8) | b[3];
  float tc = (float)t / 65536.0f;
  float pa = (float)p / 64.0f;
  if (!finiteIn(pa, PA_MIN, PA_MAX) || !finiteIn(tc, TC_MIN, TC_MAX)) return false;
  baroPa = pa;
  baroTempC = tc;
  baroLastGoodMs = millis();
  baroEverGood = true;
  return true;
}

static void readBatteries() {
  float v1 = (float)analogRead(PIN_ADC_FC_2S)  * BATT_CAL_FACTOR;
  float v2 = (float)analogRead(PIN_ADC_SOL_3S) * BATT_CAL_FACTOR;
  if (isfinite(v1)) vbatFC  = (vbatFC  <= 0.0f) ? v1 : vbatFC  + 0.2f * (v1 - vbatFC);
  if (isfinite(v2)) vbatSol = (vbatSol <= 0.0f) ? v2 : vbatSol + 0.2f * (v2 - vbatSol);
}

// ============================================================================
// 8. GNSS NMEA PARSING
// ============================================================================
static double nmeaToDeg(const char* raw, char hemi, bool isLat) {
  if (strlen(raw) < 4) return NAN;
  double v = atof(raw);
  int deg = (int)(v / 100.0);
  double mins = v - deg * 100.0;
  if (mins < 0.0 || mins >= 60.0) return NAN;
  double d = deg + mins / 60.0;
  if (d > (isLat ? 90.0 : 180.0)) return NAN;
  return (hemi == 'S' || hemi == 'W') ? -d : d;
}

static int hexVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

static void parseNmea(char* s) {
  if (s[0] != '$') { nmeaBad++; return; }
  char* star = strchr(s, '*');
  if (!star || !star[1] || !star[2]) { nmeaBad++; return; }
  uint8_t cs = 0;
  for (const char* p = s + 1; p < star; p++) cs ^= (uint8_t)*p;
  int h1 = hexVal(star[1]), h2 = hexVal(star[2]);
  if (h1 < 0 || h2 < 0 || cs != (uint8_t)((h1 << 4) | h2)) { nmeaBad++; return; }
  *star = '\0';

  char* t[16];
  uint8_t c = 0;
  t[c++] = s + 1;
  for (char* p = s + 1; *p && c < 16; p++) {
    if (*p == ',') { *p = '\0'; t[c++] = p + 1; }
  }
  nmeaGood++;
  if (c < 10 || strlen(t[0]) != 5 || strcmp(t[0] + 2, "GGA") != 0) return;

  uint8_t q    = (uint8_t)atoi(t[6]);
  uint8_t sats = (uint8_t)atoi(t[7]);
  gQuality = q;
  gSats    = sats;
  gLastGgaMs = millis();
  if (q == 0) return;
  double la = nmeaToDeg(t[2], t[3][0], true);
  double lo = nmeaToDeg(t[4], t[5][0], false);
  if (!isfinite(la) || !isfinite(lo)) return;
  gLat = la;
  gLon = lo;
  // Validate MSL altitude before accepting (atof returns 0 for empty/corrupt fields)
  float ga = (float)atof(t[9]);
  if (ga > -500.0f && ga < 9000.0f) gAlt = ga;
  strncpy(gTime, t[1], sizeof(gTime) - 1);
  gTime[sizeof(gTime) - 1] = '\0';
}

static void processGnss() {
  static char buf[128];
  static uint8_t idx = 0;
  int guard = 256;
  while (guard-- > 0 && GNSS_SERIAL.available()) {
    char c = (char)GNSS_SERIAL.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (idx > 8) { buf[idx] = '\0'; parseNmea(buf); }
      idx = 0;
    } else if (idx < sizeof(buf) - 1) {
      buf[idx++] = c;
    } else {
      idx = 0;
    }
  }
}

static bool gnssFixValid() {
  return gQuality > 0 && (millis() - gLastGgaMs) < 3000 && gLastGgaMs != 0 && gTime[0] != '\0';
}

// ============================================================================
// 9. ACTUATOR CONTROLS
// ============================================================================
static void energizeDrogueHold() {
  if (drogueFired) {
    // Drogue already released (manual EJECT or auto-deploy) — never re-energise.
    // Without this guard a reset after EJECT would re-energise on the next
    // calibration completion, firing a second back-EMF spike.
    Serial.println("[ACT] energizeDrogueHold skipped — drogueFired already set");
    return;
  }
  digitalWriteFast(PIN_SOL_DROGUE, HIGH);
  Serial.println("[ACT] Drogue solenoid ENERGIZED (latch closed)");
}

static void releaseDrogueLatch() {
  if (drogueFired) return;
  digitalWriteFast(PIN_SOL_DROGUE, LOW);
  drogueFired = true;
  beepOnce(300);
  Serial.println("[ACT] Drogue solenoid DE-ENERGIZED (latch released / drogue ejected)");
}

static void fireMain() {
  if (mainFired) return;
  digitalWriteFast(PIN_SOL_MAIN, HIGH);
  mainFireOffMs = millis() + MAIN_FIRE_DURATION_MS;
  mainFired = true;
  beepOnce(500);
  Serial.println("[ACT] Main solenoid ON — 3 s pulse");
}

static void serviceActuators() {
  uint32_t now = millis();
  if (mainFireOffMs != 0 && (int32_t)(now - mainFireOffMs) >= 0) {
    digitalWriteFast(PIN_SOL_MAIN, LOW);
    mainFireOffMs = 0;
    Serial.println("[ACT] Main solenoid OFF (3 s pulse complete)");
  }
  // End of GCS-commanded drogue test pulse — drive LOW to release
  if (droguePulseOffMs != 0 && (int32_t)(now - droguePulseOffMs) >= 0) {
    digitalWriteFast(PIN_SOL_DROGUE, LOW);
    droguePulseOffMs = 0;
    Serial.println("[ACT] Drogue test pulse complete — pin 22 LOW");
  }
  // Safety: release drogue hold if 45 s pass with no apogee detection
  if (!drogueFired && fsmState == FSM_ASCENT && (now - tLaunchMs > DROGUE_MAX_HOLD_MS)) {
    digitalWriteFast(PIN_SOL_DROGUE, LOW);
    drogueFired = true;
    Serial.println("[SAFETY] Drogue max-hold timeout — solenoid released!");
  }
}

static void setState(uint8_t s) {
  fsmState = s;
  if (s == FSM_PAD)    energizeDrogueHold();
  if (s == FSM_DEPLOY) releaseDrogueLatch();
  if (s == FSM_MAIN)   fireMain();
}

// ============================================================================
// 10. CALIBRATION & FLIGHT STATE MACHINE (runs at 100 Hz via sensorTick)
// ============================================================================
static void updateCalibration(bool imuOk, bool baroOk) {
  uint32_t now = millis();

  if (calPhase == 0) {
    if (!resumeCandidate) { calPhase = 1; return; }
    if (baroOk) {
      float agl = pressureToAlt(baroPa, savedRec.groundPa);
      if (agl > RESUME_MIN_AGL_M && agl < RESUME_MAX_AGL_M) {
        groundPa = savedRec.groundPa;
        for (uint8_t i = 0; i < 3; i++) gyroBias[i] = savedRec.gyroBias[i];
        drogueFired = savedRec.drogueFired != 0;  // restore before energizeDrogueHold
        aglFilt = aglPrev1s = maxAgl = agl;
        launchConfirmed = true;
        tLaunchMs = now - MOTOR_BURNOUT_MS - 1;
        energizeDrogueHold();   // guard inside skips if drogueFired already set
        setState(FSM_ASCENT);
        calPhase = 2;
        Serial.println("[RESUME] In-flight state restored from EEPROM");
        return;
      }
      recordSave(0);
      calPhase = 1;
    } else if (now - calPhase0Start > 2000) {
      calPhase = 1;
    }
    return;
  }

  if (calPhase == 1) {
    if (!imuOk || !baroOk) return;
    float an = sqrtf(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2]);
    bool still = fabsf(gyro[0]) < CAL_GYRO_MAX_DPS && fabsf(gyro[1]) < CAL_GYRO_MAX_DPS &&
                 fabsf(gyro[2]) < CAL_GYRO_MAX_DPS && an > CAL_ACC_MIN_MS2 && an < CAL_ACC_MAX_MS2;
    if (!still) {
      calCount = 0; calPaSum = 0;
      calGyroSum[0] = calGyroSum[1] = calGyroSum[2] = 0;
      return;
    }
    for (uint8_t i = 0; i < 3; i++) calGyroSum[i] += gyro[i];
    calPaSum += baroPa;
    if (++calCount >= CAL_SAMPLES) {
      for (uint8_t i = 0; i < 3; i++) gyroBias[i] = calGyroSum[i] / CAL_SAMPLES;
      groundPa = (float)(calPaSum / CAL_SAMPLES);
      recordSave(0);
      aglFilt = aglPrev1s = maxAgl = 0;
      calPhase = 2;
      setState(FSM_PAD);
      beepOnce(250);
      Serial.printf("[CAL] Done: groundPa=%.1f gyroBias=[%.2f %.2f %.2f]\n",
                    groundPa, gyroBias[0], gyroBias[1], gyroBias[2]);
    }
  }
}

static void updateFlight(bool imuFresh) {
  uint32_t now = millis();
  float an = sqrtf(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2]);

  // 1 Hz vertical speed — 1-second window keeps baro noise below landing threshold
  if (++tick1sCount >= 100) {
    tick1sCount = 0;
    vertSpeed  = aglFilt - aglPrev1s;
    aglPrev1s  = aglFilt;
    if (fabsf(vertSpeed) < LAND_VS_MS) { if (stillCnt < 255) stillCnt++; }
    else stillCnt = 0;
  }

  switch (fsmState) {
    case FSM_PAD:
      if (imuFresh && an > LAUNCH_ACC_MS2) launchCnt++; else launchCnt = 0;
      if (launchCnt >= LAUNCH_SAMPLES) {
        tLaunchMs = now;
        launchConfirmed = false;
        maxAgl = aglFilt;
        apogeeCnt = 0;
        launchCnt = 0;
        stillCnt  = 0;   // clear pad-time still count; movement will re-zero it properly
        setState(FSM_ASCENT);
        recordSave(1);
        Serial.println("[FSM] Launch detected — drogue LOCKED ON");
      }
      break;

    case FSM_ASCENT:
      if (aglFilt > maxAgl) maxAgl = aglFilt;
      if (!launchConfirmed) {
        if (aglFilt > LAUNCH_CONFIRM_M) {
          launchConfirmed = true;
        } else if (now - tLaunchMs > LAUNCH_CONFIRM_MS) {
          setState(FSM_PAD);
          recordSave(0);
          Serial.println("[FSM] Launch aborted — altitude not confirmed");
          break;
        }
      }
      if (launchConfirmed && (now - tLaunchMs > MOTOR_BURNOUT_MS) &&
          (maxAgl - aglFilt) > APOGEE_DROP_M) {
        if (++apogeeCnt >= APOGEE_SAMPLES) {
          tDeployMs = now;
          setState(FSM_DEPLOY);
          Serial.printf("[FSM] Apogee at %.1f m — drogue released\n", maxAgl);
        }
      } else {
        apogeeCnt = 0;
      }
      break;

    case FSM_DEPLOY:
      if (now - tDeployMs >= DEPLOY_HOLD_MS) setState(FSM_DESCENT);
      break;

    case FSM_DESCENT:
      if (aglFilt <= MAIN_DEPLOY_AGL_M && vertSpeed < -1.0f) {
        if (!REQUIRE_DROGUE_FOR_MAIN || drogueFired) {
          setState(FSM_MAIN);
          Serial.printf("[FSM] %.1f m AGL — main unlatch for 3 s\n", aglFilt);
        }
      }
      break;

    default:
      break;
  }

  // Landing confirmation (any post-apogee state)
  if ((fsmState == FSM_DEPLOY || fsmState == FSM_DESCENT || fsmState == FSM_MAIN) &&
      stillCnt >= LAND_STILL_WINDOWS && aglFilt < LAND_MAX_AGL_M) {
    setState(FSM_IMPACT);
    digitalWriteFast(PIN_SOL_DROGUE, LOW);
    digitalWriteFast(PIN_SOL_MAIN,   LOW);
    recordSave(0);
    Serial.println("[FSM] Impact confirmed — both solenoids OFF");
  }
}

static void sensorTick() {
  bool imuOk  = readImu();
  bool baroOk = readBaro();

  if (!imuOk)  imuFailures++;
  if (!baroOk) baroFailures++;
  if (imuOk && baroOk) i2cFailStreak = 0;
  else if (i2cFailStreak < 255) i2cFailStreak++;
  if (i2cFailStreak >= 5) i2cRecover();

  if (baroOk) {
    aglRaw = (calPhase == 2) ? pressureToAlt(baroPa, groundPa) : 0.0f;
    aglFilt += 0.15f * (aglRaw - aglFilt);
  }

  if (calPhase != 2) { updateCalibration(imuOk, baroOk); return; }
  if (baroOk) updateFlight(imuOk);
}

// ============================================================================
// 11. TELEMETRY FRAME BUILDERS
// ============================================================================
static void fmtOpt(char* dst, size_t n, bool ok, uint8_t dec, double v) {
  if (!ok || !isfinite(v)) { dst[0] = '\0'; return; }
  snprintf(dst, n, "%.*f", (int)dec, v);
}

// Builds exactly the 22-field ROCKET frame the GCS expects.
// Format: $TEAM_ID,ROCKET,TIMESTAMP,COUNT,ALT,PRESS,TEMP,VOLT,
//         NAVTIME,LAT,LON,NAV_ALT,SATS,AX,AY,AZ,GX,GY,GZ,FSM,SOL,NIC*XX
static size_t buildTelemetry(char* out, size_t cap, uint32_t count, uint32_t nowMs) {
  bool fix      = gnssFixValid();
  bool imuFresh = (nowMs - imuLastGoodMs) < 500 && imuEverGood;

  char lat[20], lon[20], nalt[14], ax[14], ay[14], az[14], gx[14], gy[14], gz[14];
  fmtOpt(lat,  sizeof(lat),  fix,      6, gLat);
  fmtOpt(lon,  sizeof(lon),  fix,      6, gLon);
  fmtOpt(nalt, sizeof(nalt), fix,      1, gAlt);
  fmtOpt(ax,   sizeof(ax),   imuFresh, 3, accel[0]);
  fmtOpt(ay,   sizeof(ay),   imuFresh, 3, accel[1]);
  fmtOpt(az,   sizeof(az),   imuFresh, 3, accel[2]);
  fmtOpt(gx,   sizeof(gx),   imuFresh, 2, gyro[0]);
  fmtOpt(gy,   sizeof(gy),   imuFresh, 2, gyro[1]);
  fmtOpt(gz,   sizeof(gz),   imuFresh, 2, gyro[2]);

  bool solFlag  = REPORT_FSM_FLAGS && drogueFired;
  bool mainFlag = REPORT_FSM_FLAGS && mainFired;

  char body[240];
  int n = snprintf(body, sizeof(body),
      "%s,ROCKET,%lu.%02lu,%lu,%.2f,%.2f,%.2f,%.2f,%s,%s,%s,%s,%u,%s,%s,%s,%s,%s,%s,%u,%u,%u",
      TEAM_ID,
      (unsigned long)(nowMs / 1000UL), (unsigned long)((nowMs % 1000UL) / 10UL),
      (unsigned long)count,
      (double)aglRaw, (double)(baroPa / 100.0f), (double)baroTempC, (double)vbatFC,
      fix ? gTime : "", lat, lon, nalt, (unsigned)gSats,
      ax, ay, az, gx, gy, gz,
      (unsigned)fsmState, solFlag ? 1u : 0u, mainFlag ? 1u : 0u);
  if (n <= 0 || (size_t)n >= sizeof(body)) return 0;

  int m = snprintf(out, cap, "$%s*%02X", body, (unsigned)xorChecksum(body));
  if (m <= 0 || (size_t)m >= cap) return 0;
  return (size_t)m;
}

// Short LoRa beacon (full packet is ~3 s airtime at SF10)
static size_t buildBeacon(char* out, size_t cap, uint32_t count) {
  bool fix = gnssFixValid();
  char lat[20], lon[20];
  fmtOpt(lat, sizeof(lat), fix, 6, gLat);
  fmtOpt(lon, sizeof(lon), fix, 6, gLon);
  char body[110];
  int n = snprintf(body, sizeof(body), "%s,RDF,%lu,%.0f,%s,%s,%u,%u,%.1f,%.1f",
                   TEAM_ID, (unsigned long)count, (double)aglRaw, lat, lon,
                   (unsigned)gSats, (unsigned)fsmState, (double)vbatFC, (double)vbatSol);
  if (n <= 0 || (size_t)n >= sizeof(body)) return 0;
  int m = snprintf(out, cap, "$%s*%02X", body, (unsigned)xorChecksum(body));
  if (m <= 0 || (size_t)m >= cap) return 0;
  return (size_t)m;
}

// ============================================================================
// 12. XBEE 3 API-1 ENGINE
// ============================================================================
// Low-level transmit — takes explicit destination MAC so it can send PONG to
// the exact sender without going through the RX1/RX2 round-robin.
static bool xbeeSendFrame(const uint8_t* destMac, const char* payload, size_t len) {
  if (len == 0 || len > 220) return false;
  uint8_t fid = nextFrameId;
  nextFrameId = (nextFrameId >= 255) ? 1 : nextFrameId + 1;

  uint8_t pkt[4 + 14 + 220 + 1];
  uint16_t frameLen = (uint16_t)(14 + len);
  pkt[0] = 0x7E;
  pkt[1] = (uint8_t)(frameLen >> 8);
  pkt[2] = (uint8_t)(frameLen & 0xFF);
  pkt[3] = 0x10;        // Transmit Request
  pkt[4] = fid;
  memcpy(&pkt[5], destMac, 8);
  pkt[13] = 0xFF; pkt[14] = 0xFE;  // 16-bit addr unknown
  pkt[15] = 0x00; pkt[16] = 0x00;  // broadcast radius, options
  memcpy(&pkt[17], payload, len);
  uint32_t sum = 0;
  for (size_t i = 3; i < 17 + len; i++) sum += pkt[i];
  pkt[17 + len] = (uint8_t)(0xFF - (sum & 0xFF));
  size_t total = 18 + len;

  if ((size_t)XBEE_SERIAL.availableForWrite() < total) { txDropped++; return false; }
  XBEE_SERIAL.write(pkt, total);
  txSentX++;
  sentSinceStatus++;
  return true;
}

// Round-robin telemetry send — odd packetCount → RX1, even → RX2.
static bool xbeeSend(uint8_t destIndex, const char* payload, size_t len) {
  frameIdDest[nextFrameId] = destIndex;  // map before xbeeSendFrame increments nextFrameId
  return xbeeSendFrame(destIndex == 0 ? XBEE_MAC_RX1 : XBEE_MAC_RX2, payload, len);
}

static void noteTxStatus(uint8_t id, uint8_t status) {
  uint8_t dest = frameIdDest[id] & 1;
  lastStatusCode = status;
  lastStatusMs = millis();
  sentSinceStatus = 0;
  if (status == 0) statusOk[dest]++; else statusFail[dest]++;
}

// ============================================================================
// 13. UPLINK COMMAND HANDLER
// ============================================================================
// senderMac is the 8-byte 64-bit address of the source XBee (may be NULL if the
// receive frame only carries a 16-bit address — PONG is sent to broadcast then).
static void handleCommand(const uint8_t* p, size_t n, const uint8_t* senderMac) {
  if (n == 0) return;

  // --- PING / $PING → PONG (link-test diagnostic, works at any time) ---
  bool isPing = (n >= 4 && memcmp(p, "PING",  4) == 0) ||
                (n >= 5 && memcmp(p, "$PING", 5) == 0);
  if (isPing) {
    pingsAnswered++;
    if (ENABLE_GCS_PING_BUZZ && fsmState <= FSM_PAD) beepOnce(80);
    char pong[80];
    snprintf(pong, sizeof(pong), "%s,PONG,%lu,%u,%.2f",
             TEAM_ID, (unsigned long)packetCount, (unsigned)fsmState, (double)vbatFC);
    char outPkt[96];
    snprintf(outPkt, sizeof(outPkt), "$%s*%02X\r\n", pong, (unsigned)xorChecksum(pong));
    static const uint8_t BCAST[8] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF};
    xbeeSendFrame(senderMac ? senderMac : BCAST, outPkt, strlen(outPkt));
    return;
  }

  // --- Single-character commands: strip surrounding whitespace / NUL bytes ---
  size_t a = 0, b = n;
  while (a < b && (p[a] == '\r' || p[a] == '\n' || p[a] == ' ' || p[a] == 0)) a++;
  while (b > a && (p[b-1] == '\r' || p[b-1] == '\n' || p[b-1] == ' ' || p[b-1] == 0)) b--;
  if (b - a != 1) { cmdIgnored++; return; }

  char c = (char)p[a];
  if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
  uint32_t now = millis();

  if (c == 'A') {
    // Start telemetry (absolute, not a toggle — safe to repeat from both radios)
    if (!txEnabled) { lastStatusMs = now; sentSinceStatus = 0; beepOnce(150); }
    txEnabled = true;
    cmdAccepted++; lastCmd = 'A'; lastCmdMs = now;

  } else if (c == 'B') {
    // Stop telemetry — refused mid-flight so stray characters can't silence vehicle
    if (inFlightNow() && !ALLOW_STOP_IN_FLIGHT) {
      cmdRejected++; lastCmd = 'b'; lastCmdMs = now;  // lowercase = refused
      return;
    }
    if (txEnabled) beepOnce(600);
    txEnabled = false;
    cmdAccepted++; lastCmd = 'B'; lastCmdMs = now;

  } else if (c == 'E') {
    // Unconditional GCS eject override, repeatable — fires every time.
    //
    // Ground (fsmState <= FSM_PAD): brief 150 ms HIGH pulse then LOW so each
    // button press creates a physical solenoid click the operator can hear/feel,
    // allowing the mechanism to be tested multiple times after manual re-latching.
    //
    // In flight (fsmState > FSM_PAD): skip the HIGH pulse — never briefly
    // re-engage the drogue latch mid-flight. Just confirm pin LOW immediately.
    //
    // recordSave intentionally omitted: EEPROM writes during the back-EMF
    // window risk a mid-write reset that corrupts the record.
    drogueFired = true;
    beepOnce(300);
    if (fsmState < FSM_ASCENT || fsmState > FSM_MAIN) {
      // Ground / post-flight: 150 ms HIGH pulse so the solenoid clicks every press
      digitalWriteFast(PIN_SOL_DROGUE, HIGH);
      droguePulseOffMs = now + 150;                // serviceActuators drives LOW after 150 ms
    } else {
      digitalWriteFast(PIN_SOL_DROGUE, LOW);       // mid-flight: release immediately, no re-engage
      droguePulseOffMs = 0;
    }
    if (fsmState < FSM_DEPLOY) {
      tDeployMs = now;
      fsmState  = FSM_DEPLOY;
    }
    cmdAccepted++; lastCmd = 'E'; lastCmdMs = now;
    Serial.printf("[CMD] EJECT — %s pulse, was state %u\n",
                  fsmState <= FSM_PAD ? "test" : "flight", (unsigned)fsmState);

  } else {
    cmdIgnored++;
  }
}

// ============================================================================
// 14. XBEE RECEIVE FRAME PARSER
// ============================================================================
static void xbeePoll() {
  static uint8_t  st = 0;
  static uint16_t flen = 0, fpos = 0;
  static uint8_t  fdata[128];
  static uint8_t  fsum = 0;
  int guard = 256;
  while (guard-- > 0 && XBEE_SERIAL.available()) {
    uint8_t b = (uint8_t)XBEE_SERIAL.read();
    switch (st) {
      case 0: if (b == 0x7E) st = 1; break;
      case 1: flen = (uint16_t)(b << 8); st = 2; break;
      case 2:
        flen = (uint16_t)(flen | b);
        if (flen == 0 || flen > 300) { st = (b == 0x7E) ? 1 : 0; break; }
        fpos = 0; fsum = 0; st = 3; break;
      case 3:
        if (fpos < sizeof(fdata)) fdata[fpos] = b;
        fsum += b; fpos++;
        if (fpos >= flen) st = 4;
        break;
      case 4:
        st = 0;
        if ((uint8_t)(fsum + b) != 0xFF) { xbeeRxBadCks++; break; }
        xbeeRxFrames++;
        // TX status frames
        if (fpos >= 3 && fdata[0] == 0x89) {
          noteTxStatus(fdata[1], fdata[2]);                     // 802.15.4 TX status
        } else if (fpos >= 6 && fdata[0] == 0x8B) {
          noteTxStatus(fdata[1], fdata[5]);                     // extended TX status
        // Received data frames — handle A/B/E commands and PING
        } else if (fdata[0] == 0x90 && flen >= 13 && flen <= 64) {
          // 0x90: type(1)+addr64(8)+addr16(2)+options(1) = 12 header bytes
          handleCommand(&fdata[12], flen - 12, &fdata[1]);     // sender MAC at fdata[1]
        } else if (fdata[0] == 0x80 && flen >= 12 && flen <= 64) {
          // 0x80: type(1)+addr64(8)+rssi(1)+options(1) = 11 header bytes
          handleCommand(&fdata[11], flen - 11, &fdata[1]);     // sender MAC at fdata[1]
        } else if (fdata[0] == 0x81 && flen >= 6 && flen <= 64) {
          // 0x81: type(1)+addr16(2)+rssi(1)+options(1) = 5 header bytes (no 64-bit addr)
          handleCommand(&fdata[5], flen - 5, nullptr);         // PONG uses broadcast
        }
        break;
    }
  }
}

static void xbeeWatch() {
  uint32_t now = millis();
  if (sentSinceStatus > 60 && now - lastStatusMs > 3000 && now - lastXbeeRecoverMs > 5000) {
    lastXbeeRecoverMs = now;
    xbeeRecoveries++;
    XBEE_SERIAL.end();
    XBEE_SERIAL.begin(XBEE_BAUD);
    XBEE_SERIAL.addMemoryForWrite(xbeeTxMem, sizeof(xbeeTxMem));
    XBEE_SERIAL.addMemoryForRead(xbeeRxMem,  sizeof(xbeeRxMem));
    sentSinceStatus = 0;
    lastStatusMs = now;
  }
}

// ============================================================================
// 15. REYAX RYLR998 DRIVER & BEACON
// ============================================================================
static void loraPoll() {
  static char    line[96];
  static uint8_t idx = 0;
  int guard = 128;
  while (guard-- > 0 && LORA_SERIAL.available()) {
    char c = (char)LORA_SERIAL.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[idx] = '\0';
      if (idx > 0) {
        if (loraCfgWaiting) {
          if (strncmp(line, "+OK",  3) == 0) { loraCfgWaiting = false; loraCfgStep++; }
          else if (strncmp(line, "+ERR", 4) == 0) { loraCfgWaiting = false; loraCfgStep++; loraCfgErrors++; }
        }
        if (strncmp(line, "+RCV", 4) == 0 && strstr(line, "PING") != nullptr) {
          if (ENABLE_GCS_PING_BUZZ && fsmState <= FSM_PAD) beepOnce(80);
          LORA_SERIAL.print("AT+SEND=0,4,PONG\r\n");
          pingsAnswered++;
        }
      }
      idx = 0;
    } else if (idx < sizeof(line) - 1) {
      line[idx++] = c;
    } else {
      idx = 0;
    }
  }
}

static void loraConfigure() {
  if (loraReady) return;
  uint32_t now = millis();
  if (loraCfgWaiting) {
    if (now - loraCfgSentMs > 500) { loraCfgWaiting = false; loraCfgStep++; loraCfgErrors++; }
    return;
  }
  char cmd[64];
  switch (loraCfgStep) {
    case 0: snprintf(cmd, sizeof(cmd), "AT+BAND=%lu",         (unsigned long)LORA_BAND_HZ); break;
    case 1: snprintf(cmd, sizeof(cmd), "AT+PARAMETER=%u,%u,%u,%u", LORA_SF, LORA_BW_CODE, LORA_CR_CODE, LORA_PREAMBLE); break;
    case 2: snprintf(cmd, sizeof(cmd), "AT+NETWORKID=%u",     LORA_NETWORKID); break;
    case 3: snprintf(cmd, sizeof(cmd), "AT+ADDRESS=%u",       LORA_ADDRESS); break;
    case 4: snprintf(cmd, sizeof(cmd), "AT+CRFOP=%u",         LORA_POWER_DBM); break;
    default: loraReady = true; return;
  }
  LORA_SERIAL.print(cmd);
  LORA_SERIAL.print("\r\n");
  loraCfgSentMs = now;
  loraCfgWaiting = true;
}

static void loraSendBeacon(uint32_t count) {
  if (!loraReady) return;
  char beacon[110];
  size_t n = buildBeacon(beacon, sizeof(beacon), count);
  if (n == 0) return;
  char cmd[150];
  int m = snprintf(cmd, sizeof(cmd), "AT+SEND=0,%u,%s\r\n", (unsigned)n, beacon);
  if (m <= 0 || (size_t)m >= sizeof(cmd)) return;
  if (LORA_SERIAL.availableForWrite() < m) { loraDropped++; return; }
  LORA_SERIAL.write((const uint8_t*)cmd, (size_t)m);
  loraSent++;
}

// ============================================================================
// 16. RGB LED & BUZZER PATTERN GENERATOR
// ============================================================================
static void updateIndicators() {
  uint32_t now = millis();
  bool sensorFault = (now - baroLastGoodMs > 500) || (now - imuLastGoodMs > 500);
  bool r = false, g = false, b = false, buz = false;

  if (sensorFault && calPhase == 2) {
    r = ((now / 125) & 1); buz = r;           // fast red blink + audio fault
  } else {
    switch (fsmState) {
      case FSM_BOOT:
        if (calPhase == 1) { bool t = ((now/250)&1); r = t; g = t; }  // yellow blink: calibrating
        else r = true;
        break;
      case FSM_PAD:    g = true; buz = (now % 5000) < 50; break;       // green + heartbeat
      case FSM_ASCENT: b = true; break;
      case FSM_DEPLOY: r = true; b = true; break;                       // purple: drogue released
      case FSM_DESCENT:g = true; b = true; break;                       // cyan: descent
      case FSM_MAIN:   r = true; g = true; b = true; break;             // white: main firing
      case FSM_IMPACT: g = ((now/500)&1); buz = (now%3000) < 800; break;// recovery beacon
      default: r = true; break;
    }
  }
  if (!txEnabled) { r = ((now/500)&1); g = false; b = !r; }  // telemetry stopped: red/blue
  if ((int32_t)(beepUntilMs - now) > 0) buz = true;

  digitalWriteFast(PIN_LED_R, r);
  digitalWriteFast(PIN_LED_G, g);
  digitalWriteFast(PIN_LED_B, b);
  digitalWriteFast(PIN_BUZZER, buz);
}

#if HAVE_WDT
static void wdtWarning() { if (Serial) Serial.println("[WDT] Loop stall warning!"); }
#endif

// ============================================================================
// 17. SETUP
// ============================================================================
void setup() {
  pinMode(PIN_BUZZER,     OUTPUT);
  pinMode(PIN_LED_R,      OUTPUT);
  pinMode(PIN_LED_G,      OUTPUT);
  pinMode(PIN_LED_B,      OUTPUT);
  pinMode(PIN_SOL_MAIN,   OUTPUT);
  pinMode(PIN_SOL_DROGUE, OUTPUT);
  digitalWriteFast(PIN_BUZZER,     LOW);
  digitalWriteFast(PIN_SOL_MAIN,   LOW);
  digitalWriteFast(PIN_SOL_DROGUE, LOW);
  digitalWriteFast(PIN_LED_R, HIGH);

  Serial.begin(115200);
  GNSS_SERIAL.begin(GNSS_BAUD);
  GNSS_SERIAL.addMemoryForRead(gnssRxMem, sizeof(gnssRxMem));
  XBEE_SERIAL.begin(XBEE_BAUD);
  XBEE_SERIAL.addMemoryForWrite(xbeeTxMem, sizeof(xbeeTxMem));
  XBEE_SERIAL.addMemoryForRead(xbeeRxMem,  sizeof(xbeeRxMem));
  LORA_SERIAL.begin(LORA_BAUD);
  LORA_SERIAL.addMemoryForWrite(loraTxMem, sizeof(loraTxMem));
  LORA_SERIAL.addMemoryForRead(loraRxMem,  sizeof(loraRxMem));

  analogReadResolution(12);
  analogReadAveraging(16);

  Wire.begin();
  Wire.setClock(400000);
  initSensors();
  initSdCard();

  resumeCandidate = recordLoad() && savedRec.inFlight == 1;
  calPhase0Start  = millis();
  memset(frameIdDest, 0, sizeof(frameIdDest));

  uint32_t now = millis();
  nextSensorMs = nextTxMs = now + 20;
  nextLoraMs   = now + LORA_PERIOD_MS;
  nextStatusMs = now + STATUS_PERIOD_MS;
  nextSdFlushMs= now + SD_FLUSH_PERIOD_MS;
  nextBattMs   = now;
  lastStatusMs = now;
  baroLastGoodMs = imuLastGoodMs = now;

#if HAVE_WDT
  if (ENABLE_WATCHDOG) {
    WDT_timings_t cfg;
    cfg.trigger  = 1;
    cfg.timeout  = 2;
    cfg.callback = wdtWarning;
    wdt.begin(cfg);
  }
#else
  Serial.println("[WDT] Watchdog_t4 not installed — NO watchdog");
#endif
}

// ============================================================================
// 18. MAIN LOOP
// ============================================================================
void loop() {
  feedWatchdog();
  uint32_t now = millis();

  processGnss();
  xbeePoll();
  loraPoll();
  loraConfigure();
  xbeeWatch();
  serviceActuators();

  // 100 Hz — sensor reading, FSM, SD logging
  if ((int32_t)(now - nextSensorMs) >= 0) {
    nextSensorMs += SENSOR_PERIOD_MS;
    if ((int32_t)(now - nextSensorMs) >= 0) nextSensorMs = now + SENSOR_PERIOD_MS;
    sensorTick();

    if (ENABLE_SD_LOGGING && sdActive) {
      char csv[160];
      snprintf(csv, sizeof(csv),
               "%lu,%lu,%u,%.2f,%.2f,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.6f,%.6f,%u,%u,%u\n",
               (unsigned long)now, (unsigned long)packetCount, (unsigned)fsmState,
               (double)aglRaw, (double)(baroPa / 100.0f), (double)baroTempC,
               (double)vbatFC, (double)vbatSol,
               (double)accel[0], (double)accel[1], (double)accel[2],
               (double)gyro[0],  (double)gyro[1],  (double)gyro[2],
               gLat, gLon, (unsigned)gSats,
               drogueFired ? 1u : 0u, mainFired ? 1u : 0u);
      logToSdQueue(csv);
    }
  }

  // 10 Hz battery monitoring
  if ((int32_t)(now - nextBattMs) >= 0) {
    nextBattMs += BATT_SAMPLE_MS;
    if ((int32_t)(now - nextBattMs) >= 0) nextBattMs = now + BATT_SAMPLE_MS;
    readBatteries();
  }

  // 10 Hz SD flush (bulk writes)
  if ((int32_t)(now - nextSdFlushMs) >= 0) {
    nextSdFlushMs += SD_FLUSH_PERIOD_MS;
    if ((int32_t)(now - nextSdFlushMs) >= 0) nextSdFlushMs = now + SD_FLUSH_PERIOD_MS;
    flushSdQueue();
  }

  // 20 Hz XBee alternating downlink
  // While txEnabled is false the packetCount does not advance so the GCS sees
  // a contiguous sequence when 'A' restarts telemetry.
  if ((int32_t)(now - nextTxMs) >= 0) {
    nextTxMs += TX_PERIOD_MS;
    if ((int32_t)(now - nextTxMs) >= 0) nextTxMs = now + TX_PERIOD_MS;
    size_t n = 0;
    char frame[256];
    if (txEnabled) {
      packetCount++;
      n = buildTelemetry(frame, sizeof(frame) - 3, packetCount, now);
    }
    if (n > 0) {
      frame[n++] = '\r';
      frame[n++] = '\n';
      frame[n]   = '\0';
      xbeeSend((packetCount & 1) ? 0 : 1, frame, n);  // odd → RX1, even → RX2
    }
  }

  // 1 Hz LoRa recovery beacon (always runs regardless of txEnabled)
  if ((int32_t)(now - nextLoraMs) >= 0) {
    nextLoraMs += LORA_PERIOD_MS;
    if ((int32_t)(now - nextLoraMs) >= 0) nextLoraMs = now + LORA_PERIOD_MS;
    loraSendBeacon(packetCount);
  }

  updateIndicators();

  // 1 Hz USB diagnostic line
  if ((int32_t)(now - nextStatusMs) >= 0) {
    nextStatusMs += STATUS_PERIOD_MS;
    if ((int32_t)(now - nextStatusMs) >= 0) nextStatusMs = now + STATUS_PERIOD_MS;
    if (Serial) {
      Serial.printf(
        "[STAT] t=%lus fsm=%u agl=%.1fm P=%.1fhPa 2S=%.2fV 3S=%.2fV sats=%u "
        "| tx=%lu drop=%lu okRX1=%lu okRX2=%lu fail=%lu/%lu st=0x%02X "
        "| lora=%lu ready=%u cfgErr=%u "
        "| i2cRec=%lu imuF=%lu baroF=%lu xbRec=%lu nmea=%lu/%lu id=%u%u wdt=%d "
        "| sd=%d pings=%lu "
        "| txOn=%u cmd=%c ok=%lu refused=%lu ignored=%lu\n",
        (unsigned long)(now / 1000UL), (unsigned)fsmState,
        (double)aglRaw, (double)(baroPa / 100.0f), (double)vbatFC, (double)vbatSol,
        (unsigned)gSats,
        (unsigned long)txSentX, (unsigned long)txDropped,
        (unsigned long)statusOk[0],   (unsigned long)statusOk[1],
        (unsigned long)statusFail[0], (unsigned long)statusFail[1], lastStatusCode,
        (unsigned long)loraSent, (unsigned)loraReady, (unsigned)loraCfgErrors,
        (unsigned long)i2cRecoveries, (unsigned long)imuFailures,
        (unsigned long)baroFailures,  (unsigned long)xbeeRecoveries,
        (unsigned long)nmeaGood, (unsigned long)nmeaBad,
        (unsigned)imuIdOk, (unsigned)baroIdOk, (int)HAVE_WDT,
        (int)sdActive, (unsigned long)pingsAnswered,
        (unsigned)txEnabled, lastCmd,
        (unsigned long)cmdAccepted, (unsigned long)cmdRejected, (unsigned long)cmdIgnored);
    }
  }
}
