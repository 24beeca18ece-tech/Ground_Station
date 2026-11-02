/*
  V1Rocket.ino — Teensy 4.1 Production Flight Computer Firmware
  =============================================================================
  Hardware Stack:
    - MCU: Teensy 4.1 (600 MHz Cortex-M7)
    - Primary RF: Digi XBee 3 Pro (XB3-24AST) in API 1 Mode on Serial2
    - Long-Range RF: Reyax RYLR998 (SX1262 22 dBm) on Serial8
    - GNSS: 7Semi L89HA on Serial7 (9600 baud)
    - IMU: ST LSM6DSO32 (±32g Accel, ±2000 dps Gyro) via I2C0
    - Barometer: Bosch BMP581 (100 Hz Continuous, 8x OSR) via I2C0
    - Actuators: D4184 MOSFET — Drogue Solenoid Pin 22 only
    - Storage: Onboard MicroSD via Native 4-bit SDIO (BUILTIN_SDCARD)
  Uplink commands: 'A' start TX, 'B' stop TX, 'E' EJECT drogue, PING → PONG
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
static const char TEAM_ID[]             = "2025ASI001";

// --- Feature Enable / Disable Flags ---
constexpr bool ENABLE_WATCHDOG            = true;
constexpr bool ENABLE_SD_LOGGING          = true;
constexpr bool ENABLE_EEPROM_RESUME       = true;
constexpr bool ENABLE_GCS_PING_BUZZ       = true;
constexpr bool REPORT_FSM_FLAGS           = true;
constexpr bool ALLOW_STOP_IN_FLIGHT       = false; // 'B' command refused while airborne

// --- Digi XBee 3 Pro (XB3-24AST) 802.15.4 Ground Station MAC Addresses ---
// Flight Computer Self MAC: 00 13 A2 00 42 A0 96 41 (TX on COM9)
const uint8_t XBEE_MAC_RX1[8]             = {0x00, 0x13, 0xA2, 0x00, 0x42, 0x8D, 0xCA, 0x31}; // RX  (COM3)
const uint8_t XBEE_MAC_RX2[8]             = {0x00, 0x13, 0xA2, 0x00, 0x42, 0xA0, 0x95, 0xF5}; // RX1 (COM7)

// --- Reyax RYLR998 LoRa Radio Plan (SX1262 Transceiver) ---
constexpr uint32_t LORA_BAND_HZ           = 867500000UL; // 867.500 MHz
constexpr uint8_t  LORA_SF                = 10;
constexpr uint8_t  LORA_BW_CODE           = 7;           // 125 kHz
constexpr uint8_t  LORA_CR_CODE           = 1;           // 4/5
constexpr uint8_t  LORA_PREAMBLE          = 12;
constexpr uint8_t  LORA_NETWORKID         = 6;
constexpr uint8_t  LORA_ADDRESS           = 1;
constexpr uint8_t  LORA_POWER_DBM         = 22;

// --- Cadence & Timing (ms) ---
constexpr uint32_t SENSOR_PERIOD_MS       = 10;    // 100 Hz
constexpr uint32_t TX_PERIOD_MS           = 50;    // 20 Hz XBee downlink
constexpr uint32_t LORA_PERIOD_MS         = 1000;  // 1 Hz LoRa beacon
constexpr uint32_t STATUS_PERIOD_MS       = 1000;  // 1 Hz USB diagnostics
constexpr uint32_t SD_FLUSH_PERIOD_MS     = 500;   // 2 Hz SD flush
constexpr uint32_t BATT_SAMPLE_MS         = 100;   // 10 Hz ADC

// --- Hardware Pin Allocations ---
#define GNSS_SERIAL Serial7
#define LORA_SERIAL Serial8
#define XBEE_SERIAL Serial2
constexpr uint32_t GNSS_BAUD              = 9600;
constexpr uint32_t LORA_BAUD              = 115200;
constexpr uint32_t XBEE_BAUD             = 115200;

constexpr uint8_t PIN_BUZZER              = 4;
constexpr uint8_t PIN_ADC_FC_2S           = 24;
constexpr uint8_t PIN_ADC_SOL_3S          = 25;
constexpr uint8_t PIN_LED_R               = 30;
constexpr uint8_t PIN_LED_G               = 31;
constexpr uint8_t PIN_LED_B               = 32;
constexpr uint8_t PIN_I2C_SDA             = 18;
constexpr uint8_t PIN_I2C_SCL             = 19;

constexpr uint8_t PIN_SOL_DROGUE          = 22;     // HIGH = holding latch, LOW = release/eject
constexpr uint8_t SD_CS_PIN               = BUILTIN_SDCARD;

// --- Actuator Timers ---
constexpr uint32_t DROGUE_MAX_HOLD_MS     = 45000; // failsafe: max 45 s hold in ascent

// --- Launch & Flight State Machine Thresholds ---
constexpr float    GRAVITY_EARTH          = 9.80665f;
constexpr float    LAUNCH_ACC_MS2         = 2.5f * GRAVITY_EARTH;
constexpr uint8_t  LAUNCH_SAMPLES         = 5;
constexpr float    LAUNCH_CONFIRM_M       = 10.0f;
constexpr uint32_t LAUNCH_CONFIRM_MS      = 2500;
constexpr uint32_t MOTOR_BURNOUT_MS       = 3000;
constexpr float    APOGEE_DROP_M          = 3.0f;
constexpr uint8_t  APOGEE_SAMPLES         = 6;
constexpr uint32_t DEPLOY_HOLD_MS         = 1000;
constexpr float    MAIN_DEPLOY_AGL_M      = 400.0f;
constexpr float    LAND_VS_MS             = 0.7f;
constexpr uint8_t  LAND_STILL_WINDOWS     = 3;
constexpr float    LAND_MAX_AGL_M         = 60.0f;
constexpr float    RESUME_MIN_AGL_M       = 30.0f;
constexpr float    RESUME_MAX_AGL_M       = 6000.0f;

// --- Pad Calibration Parameters ---
constexpr uint16_t CAL_SAMPLES            = 300;
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

// --- Sensor Scales & Addresses ---
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
constexpr float    BATT_CAL_FACTOR        = (3.30f / 4095.0f) * 4.30f;
constexpr float    STD_SEA_LEVEL_PA       = 101325.0f;
constexpr float    BARO_EXPONENT          = 0.190294957f;

// --- GCS Flight States ---
constexpr uint8_t FSM_BOOT                = 0;
constexpr uint8_t FSM_TEST                = 1;
constexpr uint8_t FSM_PAD                 = 2;
constexpr uint8_t FSM_ASCENT              = 3;
constexpr uint8_t FSM_DEPLOY              = 4;
constexpr uint8_t FSM_DESCENT             = 5;
constexpr uint8_t FSM_MAIN                = 6;
constexpr uint8_t FSM_IMPACT              = 7;

// ============================================================================
// 2. INTERNAL BUFFERS & GLOBAL STATE
// ============================================================================
static uint8_t xbeeTxMem[1024];
static uint8_t xbeeRxMem[512];
static uint8_t loraTxMem[512];
static uint8_t loraRxMem[256];
static uint8_t gnssRxMem[1024];

#define SD_BUF_SIZE 2048
static char sdBuffer[SD_BUF_SIZE];
static volatile int sdHead = 0, sdTail = 0;
static File logFile;
static bool sdActive = false;

// Sensor Readings & State
static float gyroBias[3]       = {0, 0, 0};
static float groundPa          = STD_SEA_LEVEL_PA;
static float accel[3]          = {0, 0, GRAVITY_EARTH};
static float gyro[3]           = {0, 0, 0};
static float baroPa            = STD_SEA_LEVEL_PA;
static float baroTempC         = 20.0f;
static float aglRaw            = 0, aglFilt = 0, aglPrev1s = 0, vertSpeed = 0, maxAgl = 0;
static float vbatFC            = 0.0f, vbatSol = 0.0f;

static bool  baroEverGood      = false, imuEverGood = false;
static uint32_t imuLastGoodMs  = 0, baroLastGoodMs = 0;
static uint8_t  i2cFailStreak  = 0;
static uint32_t i2cRecoveries  = 0, lastI2cRecoverMs = 0;
static uint32_t imuFailures    = 0, baroFailures = 0;
static bool  imuIdOk           = false, baroIdOk = false;

// GNSS
static double gLat             = 0, gLon = 0;
static float  gAlt             = 0;
static uint8_t gSats           = 0, gQuality = 0;
static char   gTime[16]        = "";
static uint32_t gLastGgaMs     = 0;
static uint32_t nmeaGood       = 0, nmeaBad = 0;

// Flight State Machine & Actuator
static uint8_t  fsmState       = FSM_BOOT;
static bool     drogueFired    = false;
static uint32_t droguePulseOffMs = 0; // non-zero: ground test pulse end timestamp
static uint32_t tLaunchMs      = 0, tDeployMs = 0;
static bool     launchConfirmed= false;
static uint8_t  launchCnt      = 0, apogeeCnt = 0, stillCnt = 0;

// Pad Calibration State
static uint8_t  calPhase       = 0;
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

// TX start/stop uplink command state
static bool     txEnabled      = true;
static uint32_t cmdAccepted    = 0, cmdRejected = 0, cmdIgnored = 0;
static char     lastCmd        = '-';
static uint32_t lastCmdMs      = 0;

// Reyax RYLR998 Configuration State Machine
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
// 3. UTILITIES & RECOVERY
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

static uint8_t xorChecksum(const char* body) {
  uint8_t c = 0;
  while (*body) c ^= (uint8_t)*body++;
  return c;
}

static void beepOnce(uint16_t ms) {
  beepUntilMs = millis() + ms;
}

static bool inFlightNow() {
  return fsmState >= FSM_ASCENT && fsmState <= FSM_MAIN;
}

static void feedWatchdog() {
#if HAVE_WDT
  if (ENABLE_WATCHDOG) wdt.feed();
#endif
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
  }
}

static void flushSdQueue() {
  if (!ENABLE_SD_LOGGING || !sdActive || !logFile) return;
  int bytesWritten = 0;
  while (sdTail != sdHead && bytesWritten < 512) {
    logFile.write(sdBuffer[sdTail]);
    sdTail = (sdTail + 1) % SD_BUF_SIZE;
    bytesWritten++;
  }
  if (bytesWritten > 0) logFile.flush();
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
    logFile.println("TIMESTAMP_MS,PKT_COUNT,STATE,AGL_M,PRESS_HPA,TEMP_C,VBAT_2S,VBAT_3S,AX,AY,AZ,GX,GY,GZ,LAT,LON,SATS,DROGUE");
    logFile.flush();
    sdActive = true;
    Serial.printf("[SD] Flight log created: %s\n", fn);
  } else {
    sdActive = false;
    Serial.println("[SD] Failed to open flight log");
  }
}

// ============================================================================
// 5. EEPROM MID-FLIGHT RESUME ARCHITECTURE
// ============================================================================
struct CalRecord {
  uint32_t magic;
  float    groundPa;
  float    gyroBias[3];
  uint8_t  inFlight;
  uint8_t  pad[3];
  uint32_t crc;
};
constexpr uint32_t CAL_MAGIC = 0x52434B54UL; // "RCKT"
static CalRecord savedRec;

static void recordSave(uint8_t inFlight) {
  if (!ENABLE_EEPROM_RESUME) return;
  CalRecord r;
  memset(&r, 0, sizeof(r));
  r.magic = CAL_MAGIC;
  r.groundPa = groundPa;
  for (uint8_t i = 0; i < 3; i++) r.gyroBias[i] = gyroBias[i];
  r.inFlight = inFlight;
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
// 6. LOW-LEVEL I2C & AUTO-UNSTICK
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

static void initSensors() {
  uint8_t id = 0;
  imuIdOk = i2cRead(LSM_ADDR, LSM_WHO_AM_I, &id, 1) && id == 0x6C;
  i2cWrite(LSM_ADDR, LSM_CTRL3_C, 0x44);
  i2cWrite(LSM_ADDR, LSM_CTRL1_XL, 0x64);
  i2cWrite(LSM_ADDR, LSM_CTRL2_G,  0x6C);
  baroIdOk = i2cRead(BMP_ADDR, BMP_CHIP_ID, &id, 1) && id == 0x50;
  i2cWrite(BMP_ADDR, BMP_OSR_CONFIG, 0x58);
  i2cWrite(BMP_ADDR, BMP_ODR_CONFIG, 0x2B);
}

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
  Wire.setTimeout(50); // must be re-applied after every Wire.begin()
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
    int16_t rg = (int16_t)((uint16_t)b[2 * i] | ((uint16_t)b[2 * i + 1] << 8));
    int16_t ra = (int16_t)((uint16_t)b[6 + 2 * i] | ((uint16_t)b[6 + 2 * i + 1] << 8));
    g[i] = (float)rg * GYRO_SCALE_DPS;
    a[i] = (float)ra * ACCEL_SCALE_MPS2;
  }
  for (uint8_t i = 0; i < 3; i++) {
    if (!finiteIn(a[i], -ACC_ABS_MAX_MS2, ACC_ABS_MAX_MS2)) return false;
    if (!finiteIn(g[i], -GYRO_ABS_MAX_DPS, GYRO_ABS_MAX_DPS)) return false;
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
  if (t & 0x800000) t |= (int32_t)0xFF000000;
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
  gQuality = q; gSats = sats; gLastGgaMs = millis();
  if (q == 0) return;
  double la = nmeaToDeg(t[2], t[3][0], true);
  double lo = nmeaToDeg(t[4], t[5][0], false);
  if (!isfinite(la) || !isfinite(lo)) return;
  gLat = la; gLon = lo;
  gAlt = (float)atof(t[9]);
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
// 9. ACTUATOR CONTROLS (DROGUE HOLD AT PAD/ASCENT, RELEASE AT APOGEE OR CMD)
// ============================================================================
static void energizeDrogueHold() {
  if (drogueFired) { Serial.println("[ACT] energizeDrogueHold skipped — already fired"); return; }
  digitalWriteFast(PIN_SOL_DROGUE, HIGH);
  Serial.println("[ACTUATOR] Drogue Solenoid ENERGIZED (latch closed on Pad/Ascent)");
}

static void releaseDrogueLatch() {
  if (drogueFired) return;
  digitalWriteFast(PIN_SOL_DROGUE, LOW);
  drogueFired = true;
  beepOnce(300);
  Serial.println("[ACTUATOR] Apogee → Drogue Solenoid DE-ENERGIZED (latch released)");
}

static void serviceActuators() {
  uint32_t now = millis();

  // Restore pin HIGH after ground-test EJECT pulse (1 s)
  if (droguePulseOffMs != 0 && (int32_t)(now - droguePulseOffMs) >= 0) {
    digitalWriteFast(PIN_SOL_DROGUE, HIGH);
    droguePulseOffMs = 0;
    Serial.println("[ACT] Drogue test pulse complete — pin HIGH (hold restored)");
  }

  // Failsafe: if apogee never detected, release hold after 45 s in ASCENT
  if (!drogueFired && fsmState == FSM_ASCENT && (now - tLaunchMs > DROGUE_MAX_HOLD_MS)) {
    digitalWriteFast(PIN_SOL_DROGUE, LOW);
    drogueFired = true;
    Serial.println("[SAFETY] Drogue max-hold timeout — solenoid released");
  }
}

static void setState(uint8_t s) {
  fsmState = s;
  if (s == FSM_PAD)    energizeDrogueHold();
  if (s == FSM_DEPLOY) releaseDrogueLatch();
  // FSM_MAIN: no main solenoid fitted — state used for telemetry only
}

static void updateCalibration(bool imuOk, bool baroOk) {
  uint32_t now = millis();
  if (calPhase == 0) {
    if (!resumeCandidate) { calPhase = 1; return; }
    if (baroOk) {
      float agl = pressureToAlt(baroPa, savedRec.groundPa);
      if (agl > RESUME_MIN_AGL_M && agl < RESUME_MAX_AGL_M) {
        groundPa = savedRec.groundPa;
        for (uint8_t i = 0; i < 3; i++) gyroBias[i] = savedRec.gyroBias[i];
        aglFilt = aglPrev1s = maxAgl = agl;
        launchConfirmed = true;
        tLaunchMs = now - MOTOR_BURNOUT_MS - 1;
        energizeDrogueHold();
        setState(FSM_ASCENT);
        calPhase = 2;
        Serial.println("[RESUME] Restored in-flight state from EEPROM!");
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
      Serial.printf("[CALIB] Tare complete: GroundPa=%.1f, GyroBias=[%.2f,%.2f,%.2f]\n",
                    groundPa, gyroBias[0], gyroBias[1], gyroBias[2]);
    }
  }
}

static void updateFlight(bool imuFresh) {
  uint32_t now = millis();
  float an = sqrtf(accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2]);

  if (++tick1sCount >= 100) {
    tick1sCount = 0;
    vertSpeed = aglFilt - aglPrev1s;
    aglPrev1s = aglFilt;
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
        setState(FSM_ASCENT);
        // recordSave(1) deferred — EEPROM write only after altitude confirmed
        Serial.println("[FSM] Launch detected! Drogue solenoid LOCKED ON.");
      }
      break;

    case FSM_ASCENT:
      if (aglFilt > maxAgl) maxAgl = aglFilt;
      if (!launchConfirmed) {
        if (aglFilt > LAUNCH_CONFIRM_M) {
          launchConfirmed = true;
          drogueFired = false; // reset so ascent hold is valid; blocks ground-test residual
          recordSave(1);       // deferred EEPROM write — now safe, no EJECT vibration risk
          Serial.printf("[FSM] Launch confirmed at %.1fm AGL.\n", aglFilt);
        } else if (now - tLaunchMs > LAUNCH_CONFIRM_MS) {
          setState(FSM_PAD);
          recordSave(0);
          Serial.println("[FSM] Launch aborted: altitude not confirmed.");
          break;
        }
      }
      if (launchConfirmed && (now - tLaunchMs > MOTOR_BURNOUT_MS) && ((maxAgl - aglFilt) > APOGEE_DROP_M)) {
        if (++apogeeCnt >= APOGEE_SAMPLES) {
          tDeployMs = now;
          setState(FSM_DEPLOY);
          Serial.printf("[FSM] Apogee confirmed at %.1fm AGL — drogue released.\n", maxAgl);
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
        setState(FSM_MAIN);
        Serial.printf("[FSM] Reached %.1fm AGL — FSM_MAIN (no actuation, solenoid removed).\n", aglFilt);
      }
      break;

    default:
      break;
  }

  // Landing confirmation
  if ((fsmState == FSM_DEPLOY || fsmState == FSM_DESCENT || fsmState == FSM_MAIN) &&
      stillCnt >= LAND_STILL_WINDOWS && aglFilt < LAND_MAX_AGL_M) {
    setState(FSM_IMPACT);
    digitalWriteFast(PIN_SOL_DROGUE, LOW); // ensure drogue de-energized at impact
    recordSave(0);
    Serial.println("[FSM] Impact confirmed.");
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
// 10. TELEMETRY PACKET BUILDERS
// ============================================================================
static void fmtOpt(char* dst, size_t n, bool ok, uint8_t dec, double v) {
  if (!ok || !isfinite(v)) { dst[0] = '\0'; return; }
  snprintf(dst, n, "%.*f", (int)dec, v);
}

static size_t buildTelemetry(char* out, size_t cap, uint32_t count, uint32_t nowMs) {
  bool fix = gnssFixValid();
  bool imuFresh = (nowMs - imuLastGoodMs) < 500 && imuEverGood;

  char lat[20], lon[20], nalt[14], ax[14], ay[14], az[14], gx[14], gy[14], gz[14];
  fmtOpt(lat,  sizeof(lat),  fix, 6, gLat);
  fmtOpt(lon,  sizeof(lon),  fix, 6, gLon);
  fmtOpt(nalt, sizeof(nalt), fix, 1, gAlt);
  fmtOpt(ax, sizeof(ax), imuFresh, 3, accel[0]);
  fmtOpt(ay, sizeof(ay), imuFresh, 3, accel[1]);
  fmtOpt(az, sizeof(az), imuFresh, 3, accel[2]);
  fmtOpt(gx, sizeof(gx), imuFresh, 2, gyro[0]);
  fmtOpt(gy, sizeof(gy), imuFresh, 2, gyro[1]);
  fmtOpt(gz, sizeof(gz), imuFresh, 2, gyro[2]);

  bool solFlag = REPORT_FSM_FLAGS && drogueFired;

  char body[240];
  int n = snprintf(body, sizeof(body),
      "%s,ROCKET,%lu.%02lu,%lu,%.2f,%.2f,%.2f,%.2f,%s,%s,%s,%s,%u,%s,%s,%s,%s,%s,%s,%u,%u,%u",
      TEAM_ID,
      (unsigned long)(nowMs / 1000UL), (unsigned long)((nowMs % 1000UL) / 10UL),
      (unsigned long)count,
      (double)aglRaw, (double)(baroPa / 100.0f), (double)baroTempC, (double)vbatFC,
      fix ? gTime : "", lat, lon, nalt, (unsigned)gSats,
      ax, ay, az, gx, gy, gz,
      (unsigned)fsmState, solFlag ? 1u : 0u, 0u);
  if (n <= 0 || (size_t)n >= sizeof(body)) return 0;

  int m = snprintf(out, cap, "$%s*%02X", body, (unsigned)xorChecksum(body));
  if (m <= 0 || (size_t)m >= cap) return 0;
  return (size_t)m;
}

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
// 11. XBEE 3 API 1 & GCS UPLINK COMMAND ENGINE
// ============================================================================
static bool xbeeSendFrame(const uint8_t* destMac, const char* payload, size_t len) {
  if (len == 0 || len > 220) return false;
  uint8_t fid = nextFrameId;
  nextFrameId = (nextFrameId >= 255) ? 1 : nextFrameId + 1;

  uint8_t pkt[4 + 14 + 220 + 1];
  uint16_t frameLen = (uint16_t)(14 + len);
  pkt[0] = 0x7E;
  pkt[1] = (uint8_t)(frameLen >> 8);
  pkt[2] = (uint8_t)(frameLen & 0xFF);
  pkt[3] = 0x10;
  pkt[4] = fid;
  memcpy(&pkt[5], destMac, 8);
  pkt[13] = 0xFF; pkt[14] = 0xFE;
  pkt[15] = 0x00; pkt[16] = 0x00;
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

static bool xbeeSend(uint8_t destIndex, const char* payload, size_t len) {
  uint8_t fid = nextFrameId;
  frameIdDest[fid] = destIndex;
  return xbeeSendFrame(destIndex == 0 ? XBEE_MAC_RX1 : XBEE_MAC_RX2, payload, len);
}

static void noteTxStatus(uint8_t id, uint8_t status) {
  uint8_t dest = frameIdDest[id] & 1;
  lastStatusCode = status;
  lastStatusMs = millis();
  sentSinceStatus = 0;
  if (status == 0) statusOk[dest]++; else statusFail[dest]++;
}

// Handles all uplink commands from either XBee receiver.
// Commands: 'A' start TX, 'B' stop TX, 'E' eject drogue, PING → PONG.
static void handleGcsCommand(const uint8_t* senderMac, const uint8_t* payload, size_t len) {
  // PING → reply with PONG directly to the sender
  bool isPing = (len >= 4 && memcmp(payload, "PING",  4) == 0) ||
                (len >= 5 && memcmp(payload, "$PING", 5) == 0) ||
                (len == 1 && payload[0] == 'P');
  if (isPing) {
    if (ENABLE_GCS_PING_BUZZ && fsmState <= FSM_PAD) beepOnce(80);
    pingsAnswered++;
    char pong[64];
    snprintf(pong, sizeof(pong), "%s,PONG,%lu,%u,%.2f",
             TEAM_ID, (unsigned long)packetCount, (unsigned)fsmState, (double)vbatFC);
    char outPkt[80];
    int n = snprintf(outPkt, sizeof(outPkt), "$%s*%02X\r\n", pong, (unsigned)xorChecksum(pong));
    if (n > 0) xbeeSendFrame(senderMac, outPkt, (size_t)n);
    return;
  }

  // Trim leading/trailing whitespace and NUL bytes
  size_t a = 0, b = len;
  while (a < b && (payload[a] == '\r' || payload[a] == '\n' || payload[a] == ' ' || payload[a] == 0)) a++;
  while (b > a && (payload[b-1] == '\r' || payload[b-1] == '\n' || payload[b-1] == ' ' || payload[b-1] == 0)) b--;
  if (b - a != 1) { cmdIgnored++; return; }

  char c = (char)payload[a];
  if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
  uint32_t now = millis();

  if (c == 'A') {
    // ROCKET TX START
    if (!txEnabled) {
      lastStatusMs = now;
      sentSinceStatus = 0;
      beepOnce(150);
    }
    txEnabled = true;
    cmdAccepted++; lastCmd = 'A'; lastCmdMs = now;
    Serial.println("[CMD] TX START received");

  } else if (c == 'B') {
    // ROCKET TX STOP — refused if airborne and ALLOW_STOP_IN_FLIGHT is false
    if (inFlightNow() && !ALLOW_STOP_IN_FLIGHT) {
      cmdRejected++; lastCmd = 'b'; lastCmdMs = now;
      return;
    }
    if (txEnabled) beepOnce(600);
    txEnabled = false;
    cmdAccepted++; lastCmd = 'B'; lastCmdMs = now;
    Serial.println("[CMD] TX STOP received");

  } else if (c == 'E') {
    // EJECT: pin 22 HIGH during all of flight; this command sets it LOW
    drogueFired = true;   // blocks energizeDrogueHold() from re-firing on PAD re-entry
    beepOnce(300);
    if (inFlightNow() && launchConfirmed) {
      // In-flight: permanent release — advance FSM to deploy
      digitalWriteFast(PIN_SOL_DROGUE, LOW);
      droguePulseOffMs = 0;
      if (fsmState < FSM_DEPLOY) { tDeployMs = now; fsmState = FSM_DEPLOY; }
    } else {
      // Ground / pad: 1 s test pulse LOW, serviceActuators restores HIGH
      digitalWriteFast(PIN_SOL_DROGUE, LOW);
      droguePulseOffMs = now + 1000;
    }
    cmdAccepted++; lastCmd = 'E'; lastCmdMs = now;
    Serial.println("[CMD] EJECT received — drogue pin LOW");

  } else {
    cmdIgnored++;
  }
}

static void xbeePoll() {
  static uint8_t  st    = 0;
  static uint16_t flen  = 0, fpos = 0;
  static uint8_t  fdata[128];
  static uint8_t  fsum  = 0;
  int guard = 512;
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
        if (fpos >= 3 && fdata[0] == 0x89) {
          noteTxStatus(fdata[1], fdata[2]);
        } else if (fpos >= 6 && fdata[0] == 0x8B) {
          noteTxStatus(fdata[1], fdata[5]);
        } else if (fpos >= 12 && fdata[0] == 0x90) {
          // ZigBee-style 64-bit receive (header = 12 bytes)
          uint8_t sender[8];
          memcpy(sender, &fdata[1], 8);
          if (flen > 12) handleGcsCommand(sender, &fdata[12], flen - 12);
        } else if (fpos >= 11 && fdata[0] == 0x80) {
          // 802.15.4 legacy 64-bit receive (header = 11: type+addr64+rssi+opts)
          uint8_t sender[8];
          memcpy(sender, &fdata[1], 8);
          if (flen > 11) handleGcsCommand(sender, &fdata[11], flen - 11);
        } else if (fpos >= 5 && fdata[0] == 0x81) {
          // 802.15.4 legacy 16-bit receive (header = 5: type+addr16+rssi+opts)
          // XBee 3 in 802.15.4 mode produces this exclusively on real hardware.
          // No 64-bit sender MAC available; reply goes to RX1 by default.
          if (flen > 5) handleGcsCommand(XBEE_MAC_RX1, &fdata[5], flen - 5);
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
    XBEE_SERIAL.addMemoryForRead(xbeeRxMem, sizeof(xbeeRxMem));
    sentSinceStatus = 0;
    lastStatusMs = now;
  }
}

// ============================================================================
// 12. REYAX RYLR998 DRIVER & BEACON
// ============================================================================
static void loraPoll() {
  static char   line[96];
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
    case 0: snprintf(cmd, sizeof(cmd), "AT+BAND=%lu", (unsigned long)LORA_BAND_HZ); break;
    case 1: snprintf(cmd, sizeof(cmd), "AT+PARAMETER=%u,%u,%u,%u", LORA_SF, LORA_BW_CODE, LORA_CR_CODE, LORA_PREAMBLE); break;
    case 2: snprintf(cmd, sizeof(cmd), "AT+NETWORKID=%u", LORA_NETWORKID); break;
    case 3: snprintf(cmd, sizeof(cmd), "AT+ADDRESS=%u",   LORA_ADDRESS); break;
    case 4: snprintf(cmd, sizeof(cmd), "AT+CRFOP=%u",     LORA_POWER_DBM); break;
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
// 13. DYNAMIC RGB LED & BUZZER PATTERN GENERATOR
// ============================================================================
static void updateIndicators() {
  uint32_t now = millis();
  bool sensorFault = (now - baroLastGoodMs > 500) || (now - imuLastGoodMs > 500);

  bool r = false, g = false, b = false, buz = false;

  // Priority 1: TX stopped — red/blue alternating
  if (!txEnabled) {
    bool t = ((now / 200) & 1);
    r = t; b = !t;
  }
  // Priority 2: Hardware Fault Alarm
  else if (sensorFault && calPhase == 2) {
    r = ((now / 125) & 1);
    buz = r;
  }
  // Priority 3: In-Flight State-by-State Progression
  else {
    switch (fsmState) {
      case FSM_BOOT:
        if (calPhase == 1) { bool t = ((now / 250) & 1); r = t; g = t; }
        else               { r = true; }
        break;
      case FSM_PAD:
        g = true;
        buz = (now % 5000) < 50;
        break;
      case FSM_ASCENT:    b = true; break;
      case FSM_DEPLOY:    r = true; b = true; break;
      case FSM_DESCENT:   g = true; b = true; break;
      case FSM_MAIN:      r = true; g = true; b = true; break;
      case FSM_IMPACT:
        g = ((now / 500) & 1);
        buz = (now % 3000) < 800;
        break;
      default:            r = true; break;
    }
  }

  if ((int32_t)(beepUntilMs - now) > 0) buz = true;

  digitalWriteFast(PIN_LED_R, r);
  digitalWriteFast(PIN_LED_G, g);
  digitalWriteFast(PIN_LED_B, b);
  digitalWriteFast(PIN_BUZZER, buz);
}

#if HAVE_WDT
static void wdtWarning() {
  if (Serial) Serial.println("WDT WARNING: Execution loop stall detected!");
}
#endif

// ============================================================================
// 14. INITIALIZATION & SETUP
// ============================================================================
void setup() {
  pinMode(PIN_BUZZER,    OUTPUT);
  pinMode(PIN_LED_R,     OUTPUT);
  pinMode(PIN_LED_G,     OUTPUT);
  pinMode(PIN_LED_B,     OUTPUT);
  pinMode(PIN_SOL_DROGUE, OUTPUT);

  digitalWriteFast(PIN_BUZZER,     LOW);
  digitalWriteFast(PIN_SOL_DROGUE, HIGH); // energized immediately on power-on; LOW only on EJECT
  digitalWriteFast(PIN_LED_R,      HIGH); // solid red on boot

  Serial.begin(115200);
  GNSS_SERIAL.begin(GNSS_BAUD);
  GNSS_SERIAL.addMemoryForRead(gnssRxMem, sizeof(gnssRxMem));
  XBEE_SERIAL.begin(XBEE_BAUD);
  XBEE_SERIAL.addMemoryForWrite(xbeeTxMem, sizeof(xbeeTxMem));
  XBEE_SERIAL.addMemoryForRead(xbeeRxMem, sizeof(xbeeRxMem));
  LORA_SERIAL.begin(LORA_BAUD);
  LORA_SERIAL.addMemoryForWrite(loraTxMem, sizeof(loraTxMem));
  LORA_SERIAL.addMemoryForRead(loraRxMem, sizeof(loraRxMem));

  analogReadResolution(12);
  analogReadAveraging(16);

  Wire.begin();
  Wire.setClock(400000);
  Wire.setTimeout(50);
  initSensors();
  initSdCard();

  resumeCandidate = recordLoad() && savedRec.inFlight == 1;
  calPhase0Start  = millis();
  memset(frameIdDest, 0, sizeof(frameIdDest));

  uint32_t now = millis();
  nextSensorMs = nextTxMs = now + 20;
  nextLoraMs   = now + LORA_PERIOD_MS;
  nextStatusMs = now + STATUS_PERIOD_MS;
  nextSdFlushMs = now + SD_FLUSH_PERIOD_MS;
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
#endif
}

// ============================================================================
// 15. MAIN LOOP ENGINE
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

  // 100 Hz Sensor Tick & Flight State Machine
  if ((int32_t)(now - nextSensorMs) >= 0) {
    nextSensorMs += SENSOR_PERIOD_MS;
    if ((int32_t)(now - nextSensorMs) >= 0) nextSensorMs = now + SENSOR_PERIOD_MS;
    sensorTick();

    if (ENABLE_SD_LOGGING && sdActive) {
      char csv[128];
      snprintf(csv, sizeof(csv),
               "%lu,%lu,%u,%.2f,%.2f,%.1f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%.1f,%.6f,%.6f,%u,%u\n",
               (unsigned long)now, (unsigned long)packetCount, (unsigned)fsmState,
               (double)aglRaw, (double)(baroPa / 100.0f), (double)baroTempC,
               (double)vbatFC, (double)vbatSol,
               (double)accel[0], (double)accel[1], (double)accel[2],
               (double)gyro[0], (double)gyro[1], (double)gyro[2],
               gLat, gLon, (unsigned)gSats,
               drogueFired ? 1u : 0u);
      logToSdQueue(csv);
    }
  }

  // 10 Hz Battery Monitoring
  if ((int32_t)(now - nextBattMs) >= 0) {
    nextBattMs += BATT_SAMPLE_MS;
    if ((int32_t)(now - nextBattMs) >= 0) nextBattMs = now + BATT_SAMPLE_MS;
    readBatteries();
  }

  // 2 Hz SD Buffer Commit
  if ((int32_t)(now - nextSdFlushMs) >= 0) {
    nextSdFlushMs += SD_FLUSH_PERIOD_MS;
    if ((int32_t)(now - nextSdFlushMs) >= 0) nextSdFlushMs = now + SD_FLUSH_PERIOD_MS;
    flushSdQueue();
  }

  // 20 Hz XBee Alternating Downlink (gated on txEnabled)
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
      xbeeSend((packetCount & 1) ? 0 : 1, frame, n);
    }
  }

  // 1 Hz LoRa Beacon
  if ((int32_t)(now - nextLoraMs) >= 0) {
    nextLoraMs += LORA_PERIOD_MS;
    if ((int32_t)(now - nextLoraMs) >= 0) nextLoraMs = now + LORA_PERIOD_MS;
    loraSendBeacon(packetCount);
  }

  updateIndicators();

  // 1 Hz Diagnostics Monitor Line
  if ((int32_t)(now - nextStatusMs) >= 0) {
    nextStatusMs += STATUS_PERIOD_MS;
    if ((int32_t)(now - nextStatusMs) >= 0) nextStatusMs = now + STATUS_PERIOD_MS;
    if (Serial) {
      Serial.printf("[STAT] t=%lus fsm=%u agl=%.1fm P=%.1fhPa 2S=%.2fV 3S=%.2fV sats=%u | "
                    "tx=%lu drop=%lu okRX1=%lu okRX2=%lu pings=%lu | lora=%lu ready=%u | "
                    "i2cRec=%lu imuF=%lu baroF=%lu xbRec=%lu nmea=%lu/%lu id=%u%u wdt=%d | "
                    "txOn=%u cmd=%c ok=%lu refused=%lu ign=%lu sd=%d\n",
                    (unsigned long)(now / 1000UL), (unsigned)fsmState, (double)aglRaw,
                    (double)(baroPa / 100.0f), (double)vbatFC, (double)vbatSol,
                    (unsigned)gSats,
                    (unsigned long)txSentX, (unsigned long)txDropped,
                    (unsigned long)statusOk[0], (unsigned long)statusOk[1],
                    (unsigned long)pingsAnswered,
                    (unsigned long)loraSent, (unsigned)loraReady,
                    (unsigned long)i2cRecoveries, (unsigned long)imuFailures,
                    (unsigned long)baroFailures, (unsigned long)xbeeRecoveries,
                    (unsigned long)nmeaGood, (unsigned long)nmeaBad,
                    (unsigned)imuIdOk, (unsigned)baroIdOk, (int)HAVE_WDT,
                    (unsigned)txEnabled, lastCmd,
                    (unsigned long)cmdAccepted, (unsigned long)cmdRejected,
                    (unsigned long)cmdIgnored, (int)sdActive);
    }
  }
}
