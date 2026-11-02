/*
  Rocket_Telemetry_TX.ino      Teensy 4.1 rocket flight computer, telemetry TX
  =========================================================================
  Rewritten from 5_tx_test.ino so that every byte matches what the team's
  ground station (telemetry_packet.py / serial_worker.py / packet_merger.py)
  accepts.

  WHAT THE GROUND STATION EXPECTS (checked against its source)
    Frame   $2025ASI001,ROCKET,<18 common fields>,<SOLENOID>,<NICHROME>*XX
            22 fields, XOR checksum over everything between '$' and '*',
            two UPPERCASE hex digits, XBee frames end with CR LF.
    Units   TIMESTAMP s since boot | ALTITUDE m above pad | PRESSURE hPa
            TEMP degC | VOLTAGE V | NAV_TIME hhmmss.ss | LAT/LON deg
            NAV_ALT m MSL | ACC m/s2 | GYRO deg/s | FSM_STATE 0..7
    FSM     0 BOOT 1 TEST_MODE 2 LAUNCH_PAD 3 ASCENT 4 DEPLOY 5 DESCENT
            6 AEROBRAKE_RELEASE 7 IMPACT
    Link    802.15.4 API mode (AP=1), 0x10 Transmit Request, 64-bit dest.
            The GCS merges two receivers by PACKET_COUNT, so packets go to
            RX1 when the count is odd and to RX2 when it is even.

  FLIGHT SAFETY / RESILIENCE
    * hardware watchdog (Watchdog_t4, 2 s), fed only from the main loop
    * every sensor read is non blocking and checked; I2C bus unstick and
      sensor re-init after repeated failures
    * pressure, temperature and IMU values are range checked before use
    * pad calibration runs in the background while BOOT packets already flow
    * ground pressure + gyro bias are saved to EEPROM, so a mid flight reset
      resumes the flight instead of re-calibrating in the air
    * XBee TX status (0x89 / 0x8B) is tracked per receiver, UART is
      re-initialised if the radio stops answering
    * XBee uplink commands: 'A' start / 'B' stop telemetry (see section 10)
    * the TX path never blocks: if the UART buffer is full the packet is
      counted as dropped and the loop carries on

  Board: Teensy 4.1.  Library: Watchdog_t4 (Library Manager / GitHub tonton81).
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
// 1. CONFIGURATION (change only here)
// ============================================================================
static const char TEAM_ID[] = "2025ASI001";   // GCS default team id

// Serial ports
#define GNSS_SERIAL Serial7   // RX7 pin 28, TX7 pin 29, NMEA 9600
#define LORA_SERIAL Serial8   // RX8 pin 34, TX8 pin 35, 115200
#define XBEE_SERIAL Serial2   // RX2 pin 7,  TX2 pin 8,  115200 (ATBD=7)
constexpr uint32_t GNSS_BAUD = 9600;
constexpr uint32_t LORA_BAUD = 115200;
constexpr uint32_t XBEE_BAUD = 115200;

// GPIO
constexpr uint8_t PIN_BUZZER   = 4;
constexpr uint8_t PIN_ADC_BATT    = 24;  // A10, 33k / 10k divider from 2S pack
constexpr uint8_t PIN_ADC_SOL_3S  = 25;  // A11, 3S solenoid battery monitor
constexpr uint8_t SD_CS_PIN       = BUILTIN_SDCARD;
constexpr uint8_t PIN_LED_R    = 30;
constexpr uint8_t PIN_LED_G    = 31;
constexpr uint8_t PIN_LED_B    = 32;
constexpr uint8_t PIN_I2C_SDA  = 18;   // Wire
constexpr uint8_t PIN_I2C_SCL  = 19;
constexpr uint8_t PIN_SOL_DROGUE = 22;  // D4184 #2 — drogue eject solenoid
constexpr uint8_t PIN_SOL_MAIN   = 23;  // D4184 #1 — main chute solenoid
constexpr uint32_t MAIN_FIRE_DURATION_MS = 3000;  // 3 s main solenoid pulse

// Receivers. GCS RX1 gets odd packet counts, RX2 gets even packet counts.
// CONFIRM these against the 64-bit address printed on each ground XBee.
const uint8_t XBEE_MAC_RX1[8] = {0x00, 0x13, 0xA2, 0x00, 0x42, 0xA0, 0x95, 0xF5}; // RX1 (0013A20042A095F5)
const uint8_t XBEE_MAC_RX2[8] = {0x00, 0x13, 0xA2, 0x00, 0x42, 0xA0, 0x96, 0x41}; // RX2 (0013A20042A09641)

// Rates
constexpr uint32_t SENSOR_PERIOD_MS = 10;     // 100 Hz sensing + state machine
constexpr uint32_t TX_PERIOD_MS     = 50;     // 20 Hz telemetry
constexpr uint32_t LORA_PERIOD_MS   = 1000;   // 1 Hz recovery beacon
constexpr uint32_t STATUS_PERIOD_MS = 1000;   // USB debug line
constexpr uint32_t SD_FLUSH_PERIOD_MS = 500;  // 2 Hz SD buffer commit

constexpr uint32_t DROGUE_MAX_HOLD_MS   = 45000; // failsafe: release drogue if held 45 s in ASCENT
constexpr bool     REQUIRE_DROGUE_FOR_MAIN = true; // main deploys only after drogue has fired

// LoRa radio plan (REYAX RYLR896).  The module's factory band is 915 MHz,
// which is NOT legal in India, so the band must be set.  Receiver must match.
constexpr uint32_t LORA_BAND_HZ   = 867500000UL;   // 867.500 MHz
constexpr uint8_t  LORA_SF        = 10;
constexpr uint8_t  LORA_BW_CODE   = 7;             // 7 = 125 kHz
constexpr uint8_t  LORA_CR_CODE   = 1;             // 1 = 4/5
constexpr uint8_t  LORA_PREAMBLE  = 12;
constexpr uint8_t  LORA_NETWORKID = 6;
constexpr uint8_t  LORA_ADDRESS   = 1;
constexpr uint8_t  LORA_POWER_DBM = 15;            // RYLR896 maximum

// The tx test has no pyro outputs.  false: SOLENOID/NICHROME fields are 0
// (honest: nothing fired).  true: the fields follow the FSM so the GCS
// recovery lights can be exercised on the bench.
constexpr bool REPORT_EVENT_FLAGS_FROM_FSM = true;

// Uplink start/stop commands (typed in XCTU, or sent by anything on the PAN).
// 'A' = start XBee telemetry, 'B' = stop it. Both are ABSOLUTE, not toggles, so
// sending the same command from RX1 and RX2 (or twice) is harmless. The source
// address is not checked: whichever ground radio sends it, it works.
// Safety: B is refused between launch and landing so a stray character can
// never silence the vehicle in flight. The LoRa recovery beacon never stops.
constexpr bool ALLOW_STOP_IN_FLIGHT = false;

// Sensor registers
constexpr uint8_t LSM_ADDR        = 0x6A;
constexpr uint8_t LSM_WHO_AM_I    = 0x0F;
constexpr uint8_t LSM_CTRL1_XL    = 0x10;
constexpr uint8_t LSM_CTRL2_G     = 0x11;
constexpr uint8_t LSM_CTRL3_C     = 0x12;
constexpr uint8_t LSM_OUTX_L_G    = 0x22;   // 12 bytes: gyro XYZ then accel XYZ
constexpr uint8_t BMP_ADDR        = 0x47;
constexpr uint8_t BMP_CHIP_ID     = 0x01;
constexpr uint8_t BMP_TEMP_DATA0  = 0x1D;   // 6 bytes: temp XLSB..MSB, press XLSB..MSB
constexpr uint8_t BMP_OSR_CONFIG  = 0x36;
constexpr uint8_t BMP_ODR_CONFIG  = 0x37;

// Physics
constexpr float GRAVITY_EARTH    = 9.80665f;
constexpr float ACCEL_SCALE_MPS2 = (32.0f / 32768.0f) * GRAVITY_EARTH;  // +/-32 g
constexpr float GYRO_SCALE_DPS   = 0.070f;                              // 2000 dps
constexpr float BATT_CAL_FACTOR  = (3.30f / 4095.0f) * 4.30f;           // 33k/10k
constexpr float STD_SEA_LEVEL_PA = 101325.0f;
constexpr float BARO_EXPONENT    = 0.190294957f;

// Plausibility windows (anything outside is treated as a failed read)
constexpr float PA_MIN = 30000.0f,  PA_MAX = 110000.0f;
constexpr float TC_MIN = -60.0f,    TC_MAX = 100.0f;
constexpr float ACC_ABS_MAX_MS2 = 33.0f * GRAVITY_EARTH;
constexpr float GYRO_ABS_MAX_DPS = 2100.0f;

// GCS flight states
constexpr uint8_t FSM_BOOT = 0, FSM_TEST = 1, FSM_PAD = 2, FSM_ASCENT = 3,
                  FSM_DEPLOY = 4, FSM_DESCENT = 5, FSM_MAIN = 6, FSM_IMPACT = 7;

// State machine thresholds
constexpr float    LAUNCH_ACC_MS2     = 3.0f * GRAVITY_EARTH;  // 3 g
constexpr uint8_t  LAUNCH_SAMPLES     = 5;       // 50 ms above threshold
constexpr float    LAUNCH_CONFIRM_M   = 10.0f;   // must gain 10 m ...
constexpr uint32_t LAUNCH_CONFIRM_MS  = 2500;    // ... within 2.5 s or abort
constexpr uint32_t APOGEE_LOCKOUT_MS  = 1000;
constexpr float    APOGEE_DROP_M      = 3.0f;
constexpr uint8_t  APOGEE_SAMPLES     = 5;
constexpr uint32_t DEPLOY_HOLD_MS     = 1000;
constexpr float    MAIN_DEPLOY_AGL_M  = 400.0f;      // per CDR
constexpr float    LAND_VS_MS         = 0.7f;   // m/s
constexpr uint8_t  LAND_STILL_WINDOWS = 3;      // consecutive 1 s windows
constexpr float    LAND_MAX_AGL_M     = 60.0f;
constexpr float    RESUME_MIN_AGL_M   = 30.0f;
constexpr float    RESUME_MAX_AGL_M   = 6000.0f;

// Calibration
constexpr uint16_t CAL_SAMPLES     = 300;            // 3 s of stillness
constexpr float    CAL_GYRO_MAX_DPS = 15.0f;
constexpr float    CAL_ACC_MIN_MS2 = 0.7f * GRAVITY_EARTH;
constexpr float    CAL_ACC_MAX_MS2 = 1.3f * GRAVITY_EARTH;

// ============================================================================
// 2. BUFFERS AND GLOBAL STATE
// ============================================================================
static uint8_t xbeeTxMem[1024];
static uint8_t xbeeRxMem[512];
static uint8_t loraTxMem[512];
static uint8_t loraRxMem[256];
static uint8_t gnssRxMem[1024];

// SD card
#define SD_BUF_SIZE 2048
static char  sdBuffer[SD_BUF_SIZE];
static int   sdHead = 0, sdTail = 0;
static File  logFile;
static bool  sdActive = false;

// Sensor values (last good)
static float gyroBias[3] = {0, 0, 0};
static float groundPa = STD_SEA_LEVEL_PA;
static float accel[3] = {0, 0, GRAVITY_EARTH};
static float gyro[3]  = {0, 0, 0};
static float baroPa = STD_SEA_LEVEL_PA, baroTempC = 20.0f;
static float aglRaw = 0, aglFilt = 0, aglPrev1s = 0, vertSpeed = 0, maxAgl = 0;
static float vbat = 0, vbatSol = 0;
static bool  baroEverGood = false, imuEverGood = false;
static uint32_t imuLastGoodMs = 0, baroLastGoodMs = 0;
static uint8_t i2cFailStreak = 0;
static uint32_t i2cRecoveries = 0, lastI2cRecoverMs = 0;
static uint32_t imuFailures = 0, baroFailures = 0;
static bool imuIdOk = false, baroIdOk = false;

// GNSS
static double gLat = 0, gLon = 0;
static float  gAlt = 0;
static uint8_t gSats = 0, gQuality = 0;
static char   gTime[16] = "";
static uint32_t gLastGgaMs = 0;
static uint32_t nmeaGood = 0, nmeaBad = 0;

// Flight state machine
static uint8_t fsmState = FSM_BOOT;
static bool    flagSolenoid = false, flagNichrome = false;
static bool    drogueFired  = false, mainFired    = false;
static uint32_t droguePulseOffMs = 0, mainFireOffMs = 0;
static uint32_t tLaunchMs = 0, tDeployMs = 0;
static bool    launchConfirmed = false;
static uint8_t launchCnt = 0, apogeeCnt = 0, stillCnt = 0;

// Calibration
static uint8_t  calPhase = 0;            // 0 wait/resume check, 1 collect, 2 done
static uint16_t calCount = 0;
static float    calGyroSum[3] = {0, 0, 0};
static double   calPaSum = 0;
static uint32_t calPhase0Start = 0;
static bool     resumeCandidate = false;

// Telemetry counters
static uint32_t packetCount = 0;
static uint32_t txDropped = 0, txSentX = 0, loraSent = 0, loraDropped = 0;
static uint32_t statusOk[2] = {0, 0}, statusFail[2] = {0, 0};
static uint8_t  lastStatusCode = 0;
static uint8_t  frameIdDest[256];
static uint8_t  nextFrameId = 1;
static uint32_t lastStatusMs = 0, sentSinceStatus = 0, xbeeRecoveries = 0, lastXbeeRecoverMs = 0;
static uint32_t xbeeRxFrames = 0, xbeeRxBadCks = 0;

// Uplink command state. Telemetry always starts ON after any reset, so a
// watchdog reset in flight can never leave the vehicle silent.
static bool     txEnabled = true;
static uint32_t cmdAccepted = 0, cmdRejected = 0, cmdIgnored = 0, lastCmdMs = 0;
static char     lastCmd = '-';

// LoRa configuration state machine
static uint8_t  loraCfgStep = 0;
static bool     loraReady = false;
static uint32_t loraCfgSentMs = 0;
static bool     loraCfgWaiting = false;
static uint8_t  loraCfgErrors = 0;

// Scheduling
static uint32_t nextSensorMs = 0, nextTxMs = 0, nextLoraMs = 0, nextStatusMs = 0;
static uint32_t nextBattMs = 0, nextSdFlushMs = 0, tick1sCount = 0;

// ============================================================================
// 3. SMALL HELPERS
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

// XOR of every byte between '$' and '*' (exactly what the GCS verifies).
static uint8_t xorChecksum(const char* body) {
  uint8_t c = 0;
  while (*body) c ^= (uint8_t)*body++;
  return c;
}

// ============================================================================
// 4. EEPROM RECORD (survives a reset in flight)
// ============================================================================
struct CalRecord {
  uint32_t magic;
  float    groundPa;
  float    gyroBias[3];
  uint8_t  inFlight;
  uint8_t  pad[3];
  uint32_t crc;
};
constexpr uint32_t CAL_MAGIC = 0x52434B54UL;   // "RCKT"
static CalRecord savedRec;

static void recordSave(uint8_t inFlight) {
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
// 4b. SD CARD LOGGING
// ============================================================================
static void logToSdQueue(const char* str) {
  if (!sdActive) return;
  for (size_t i = 0; str[i]; i++) {
    int next = (sdHead + 1) % SD_BUF_SIZE;
    if (next != sdTail) { sdBuffer[sdHead] = str[i]; sdHead = next; }
  }
}

static void flushSdQueue() {
  if (!sdActive || !logFile) return;
  int written = 0;
  while (sdTail != sdHead && written < 512) {
    logFile.write(sdBuffer[sdTail]);
    sdTail = (sdTail + 1) % SD_BUF_SIZE;
    written++;
  }
  if (written > 0) logFile.flush();
}

static void initSdCard() {
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("[SD] Card not found");
    return;
  }
  uint16_t fltCount = 0;
  EEPROM.get(100, fltCount);
  fltCount++;
  EEPROM.put(100, fltCount);
  char fn[32];
  snprintf(fn, sizeof(fn), "FLT_%03u.CSV", fltCount);
  logFile = SD.open(fn, FILE_WRITE);
  if (logFile) {
    logFile.println("MS,PKT,FSM,AGL_M,PRESS_HPA,TEMP_C,VBAT_2S,VBAT_3S,AX,AY,AZ,GX,GY,GZ,LAT,LON,SATS,DROGUE,MAIN");
    logFile.flush();
    sdActive = true;
    Serial.printf("[SD] Logging to %s\n", fn);
  }
}

// ============================================================================
// 5. I2C LAYER WITH RECOVERY
// ============================================================================
static bool i2cWrite(uint8_t dev, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(dev);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

// Read n bytes starting at reg. Bytes are consumed in order, one statement
// each (the old sketch combined two Wire.read() calls in one expression, whose
// evaluation order C++ does not define).
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
  imuIdOk  = i2cRead(LSM_ADDR, LSM_WHO_AM_I, &id, 1) && id == 0x6C;
  // LSM6DSO32: BDU on (no torn reads), address auto increment on
  i2cWrite(LSM_ADDR, LSM_CTRL3_C, 0x44);
  // 416 Hz, +/-32 g accel ; 416 Hz, +/-2000 dps gyro
  i2cWrite(LSM_ADDR, LSM_CTRL1_XL, 0x64);
  i2cWrite(LSM_ADDR, LSM_CTRL2_G,  0x6C);
  baroIdOk = i2cRead(BMP_ADDR, BMP_CHIP_ID, &id, 1) && id == 0x50;
  // BMP581: 8x pressure oversampling, continuous mode
  i2cWrite(BMP_ADDR, BMP_OSR_CONFIG, 0x58);
  i2cWrite(BMP_ADDR, BMP_ODR_CONFIG, 0x2B);
}

// A slave that lost a clock edge can hold SDA low forever. Release the pins,
// clock SCL up to 9 times until SDA frees, send a STOP, then restart Wire.
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
  Wire.setTimeout(50);   // must be re-applied after every Wire.begin()
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
// 6. SENSOR ACQUISITION (non blocking, validated)
// ============================================================================
static bool readImu() {
  uint8_t b[12];
  if (!i2cRead(LSM_ADDR, LSM_OUTX_L_G, b, 12)) return false;
  float g[3], a[3];
  for (uint8_t i = 0; i < 3; i++) {
    int16_t rg = (int16_t)((uint16_t)b[2 * i] | ((uint16_t)b[2 * i + 1] << 8));
    int16_t ra = (int16_t)((uint16_t)b[6 + 2 * i] | ((uint16_t)b[6 + 2 * i + 1] << 8));
    g[i] = (float)rg * GYRO_SCALE_DPS;           // bias removed after calibration
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
  if (t & 0x800000) t |= (int32_t)0xFF000000;   // 24 bit signed
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

static void readBattery() {
  float v1 = (float)analogRead(PIN_ADC_BATT)   * BATT_CAL_FACTOR;
  float v2 = (float)analogRead(PIN_ADC_SOL_3S) * BATT_CAL_FACTOR;
  if (isfinite(v1)) vbat    = (vbat    <= 0.0f) ? v1 : vbat    + 0.2f * (v1 - vbat);
  if (isfinite(v2)) vbatSol = (vbatSol <= 0.0f) ? v2 : vbatSol + 0.2f * (v2 - vbatSol);
}

// ============================================================================
// 7. GNSS (NMEA, checksum validated)
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
  // talker may be GP, GN, GA, GL, GB, GI ...: match the sentence type only
  if (c < 10 || strlen(t[0]) != 5 || strcmp(t[0] + 2, "GGA") != 0) return;

  uint8_t q = (uint8_t)atoi(t[6]);
  uint8_t sats = (uint8_t)atoi(t[7]);
  gQuality = q;
  gSats = sats;
  gLastGgaMs = millis();
  if (q == 0) return;
  double la = nmeaToDeg(t[2], t[3][0], true);
  double lo = nmeaToDeg(t[4], t[5][0], false);
  if (!isfinite(la) || !isfinite(lo)) return;
  gLat = la;
  gLon = lo;
  gAlt = (float)atof(t[9]);
  strncpy(gTime, t[1], sizeof(gTime) - 1);
  gTime[sizeof(gTime) - 1] = '\0';
}

static void processGnss() {
  static char buf[128];
  static uint8_t idx = 0;
  int guard = 256;                       // bounded work per call
  while (guard-- > 0 && GNSS_SERIAL.available()) {
    char c = (char)GNSS_SERIAL.read();
    if (c == '\r') continue;
    if (c == '\n') {
      if (idx > 8) { buf[idx] = '\0'; parseNmea(buf); }
      idx = 0;
    } else if (idx < sizeof(buf) - 1) {
      buf[idx++] = c;
    } else {
      idx = 0;                           // overlong line: discard
    }
  }
}

static bool gnssFixValid() {
  return gQuality > 0 && (millis() - gLastGgaMs) < 3000 && gLastGgaMs != 0
         && gTime[0] != '\0';
}

// ============================================================================
// 8. CALIBRATION AND FLIGHT STATE MACHINE (runs at 100 Hz)
// ============================================================================
static void energizeDrogueHold() {
  if (drogueFired) { Serial.println("[ACT] energizeDrogueHold skipped — already fired"); return; }
  digitalWriteFast(PIN_SOL_DROGUE, HIGH);
  Serial.println("[ACT] Drogue solenoid ENERGIZED (latch closed)");
}

static void releaseDrogueLatch() {
  if (drogueFired) return;
  digitalWriteFast(PIN_SOL_DROGUE, LOW);
  drogueFired  = true;
  flagSolenoid = true;
  beepOnce(300);
  Serial.println("[ACT] Drogue solenoid DE-ENERGIZED (latch released)");
}

static void fireMain() {
  if (mainFired) return;
  digitalWriteFast(PIN_SOL_MAIN, LOW);
  mainFireOffMs = millis() + MAIN_FIRE_DURATION_MS;
  mainFired    = true;
  flagNichrome = true;
  beepOnce(500);
  Serial.println("[ACT] Main solenoid RELEASED — 3 s LOW pulse");
}

static void serviceActuators() {
  uint32_t now = millis();
  if (mainFireOffMs != 0 && (int32_t)(now - mainFireOffMs) >= 0) {
    digitalWriteFast(PIN_SOL_MAIN, LOW);
    mainFireOffMs = 0;
    Serial.println("[ACT] Main solenoid OFF");
  }
  if (droguePulseOffMs != 0 && (int32_t)(now - droguePulseOffMs) >= 0) {
    digitalWriteFast(PIN_SOL_DROGUE, HIGH);
    droguePulseOffMs = 0;
    Serial.println("[ACT] Drogue test pulse complete — pin HIGH (hold restored)");
  }
  // Failsafe: if apogee never detected, release drogue hold after 45 s in ASCENT
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
  if (s == FSM_MAIN)   fireMain();
}

static uint32_t beepUntilMs = 0;
static void beepOnce(uint16_t ms) {      // non blocking: updateIndicators() ends it
  beepUntilMs = millis() + ms;
}

static void feedWatchdog() {
#if HAVE_WDT
  wdt.feed();
#endif
}

static void updateCalibration(bool imuOk, bool baroOk) {
  uint32_t now = millis();

  if (calPhase == 0) {
    // Mid flight reset check: wait up to 2 s for a first good pressure.
    if (!resumeCandidate) { calPhase = 1; return; }
    if (baroOk) {
      float agl = pressureToAlt(baroPa, savedRec.groundPa);
      if (agl > RESUME_MIN_AGL_M && agl < RESUME_MAX_AGL_M) {
        groundPa = savedRec.groundPa;
        for (uint8_t i = 0; i < 3; i++) gyroBias[i] = savedRec.gyroBias[i];
        aglFilt = aglPrev1s = maxAgl = agl;
        launchConfirmed = true;
        tLaunchMs = now - APOGEE_LOCKOUT_MS - 1;
        setState(FSM_ASCENT);            // apogee logic sorts out the rest
        calPhase = 2;
        return;
      }
      recordSave(0);                     // stale flag: we are on the ground
      calPhase = 1;
    } else if (now - calPhase0Start > 2000) {
      calPhase = 1;
    }
    return;
  }

  if (calPhase == 1) {
    if (!imuOk || !baroOk) return;
    // Raw gyro (bias not yet known) and accel magnitude must show stillness.
    float an = sqrtf(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]);
    bool still = fabsf(gyro[0]) < CAL_GYRO_MAX_DPS && fabsf(gyro[1]) < CAL_GYRO_MAX_DPS &&
                 fabsf(gyro[2]) < CAL_GYRO_MAX_DPS && an > CAL_ACC_MIN_MS2 && an < CAL_ACC_MAX_MS2;
    if (!still) {                        // being handled: start the average again
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
      setState(FSM_PAD);
      calPhase = 2;
      beepOnce(150);
    }
  }
}

static void updateFlight(bool imuFresh) {
  uint32_t now = millis();
  float an = sqrtf(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]);

  // 1 Hz vertical speed from the filtered barometric altitude. A 1 s window
  // keeps barometer noise (about 0.1 m) well below the 0.7 m/s stillness limit.
  if (++tick1sCount >= 100) {
    tick1sCount = 0;
    vertSpeed = aglFilt - aglPrev1s;                  // m per 1 s = m/s
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
        // recordSave(1) deferred — EEPROM write happens only after altitude confirmed
      }
      break;

    case FSM_ASCENT:
      if (aglFilt > maxAgl) maxAgl = aglFilt;
      if (!launchConfirmed) {
        if (aglFilt > LAUNCH_CONFIRM_M) { launchConfirmed = true; drogueFired = false; recordSave(1); }
        else if (now - tLaunchMs > LAUNCH_CONFIRM_MS) {   // false trigger (a knock)
          setState(FSM_PAD);
          recordSave(0);
          break;
        }
      }
      if (launchConfirmed && now - tLaunchMs > APOGEE_LOCKOUT_MS &&
          (maxAgl - aglFilt) > APOGEE_DROP_M) {
        if (++apogeeCnt >= APOGEE_SAMPLES) { tDeployMs = now; setState(FSM_DEPLOY); }
      } else {
        apogeeCnt = 0;
      }
      break;

    case FSM_DEPLOY:
      if (now - tDeployMs >= DEPLOY_HOLD_MS) setState(FSM_DESCENT);
      break;

    case FSM_DESCENT:
      if (aglFilt <= MAIN_DEPLOY_AGL_M && vertSpeed < -1.0f) {
        if (!REQUIRE_DROGUE_FOR_MAIN || drogueFired) setState(FSM_MAIN);
      }
      break;

    default:
      break;
  }

  // Landing: held still, low, after the apogee event
  if ((fsmState == FSM_DEPLOY || fsmState == FSM_DESCENT || fsmState == FSM_MAIN) &&
      stillCnt >= LAND_STILL_WINDOWS && aglFilt < LAND_MAX_AGL_M) {
    setState(FSM_IMPACT);
    recordSave(0);
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
    // Before calibration there is no pad reference: report 0 m, not a number
    // referenced to standard sea level (which would show ~300 m at Jammu).
    aglRaw = (calPhase == 2) ? pressureToAlt(baroPa, groundPa) : 0.0f;
    aglFilt += 0.15f * (aglRaw - aglFilt);
  }

  if (calPhase != 2) { updateCalibration(imuOk, baroOk); return; }
  if (baroOk) updateFlight(imuOk);
}

// ============================================================================
// 9. TELEMETRY FRAME (exactly the GCS ROCKET layout)
// ============================================================================
static void fmtOpt(char* dst, size_t n, bool ok, uint8_t dec, double v) {
  if (!ok || !isfinite(v)) { dst[0] = '\0'; return; }
  snprintf(dst, n, "%.*f", (int)dec, v);
}

// Builds "$body*XX" (no line ending) into out. Returns its length.
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

  bool sol = REPORT_EVENT_FLAGS_FROM_FSM && flagSolenoid;
  bool nic = REPORT_EVENT_FLAGS_FROM_FSM && flagNichrome;

  char body[220];
  int n = snprintf(body, sizeof(body),
      "%s,ROCKET,%lu.%02lu,%lu,%.2f,%.2f,%.2f,%.2f,%s,%s,%s,%s,%u,%s,%s,%s,%s,%s,%s,%u,%u,%u",
      TEAM_ID,
      (unsigned long)(nowMs / 1000UL), (unsigned long)((nowMs % 1000UL) / 10UL),
      (unsigned long)count,
      (double)aglRaw, (double)(baroPa / 100.0f), (double)baroTempC, (double)vbat,
      fix ? gTime : "", lat, lon, nalt, (unsigned)gSats,
      ax, ay, az, gx, gy, gz,
      (unsigned)fsmState, sol ? 1u : 0u, nic ? 1u : 0u);
  if (n <= 0 || (size_t)n >= sizeof(body)) return 0;

  int m = snprintf(out, cap, "$%s*%02X", body, (unsigned)xorChecksum(body));
  if (m <= 0 || (size_t)m >= cap) return 0;
  return (size_t)m;
}

// Short beacon for the LoRa link (full frame would take ~3 s of airtime at SF10)
static size_t buildBeacon(char* out, size_t cap, uint32_t count) {
  bool fix = gnssFixValid();
  char lat[20], lon[20];
  fmtOpt(lat, sizeof(lat), fix, 6, gLat);
  fmtOpt(lon, sizeof(lon), fix, 6, gLon);
  char body[96];
  int n = snprintf(body, sizeof(body), "%s,RDF,%lu,%.0f,%s,%s,%u,%u,%.1f",
                   TEAM_ID, (unsigned long)count, (double)aglRaw, lat, lon,
                   (unsigned)gSats, (unsigned)fsmState, (double)vbat);
  if (n <= 0 || (size_t)n >= sizeof(body)) return 0;
  int m = snprintf(out, cap, "$%s*%02X", body, (unsigned)xorChecksum(body));
  if (m <= 0 || (size_t)m >= cap) return 0;
  return (size_t)m;
}

// ============================================================================
// 10. XBEE 802.15.4 API LAYER (AP=1, raw frames, no mesh)
// ============================================================================
// 0x10 Transmit Request with a non zero frame id so the radio answers with a
// TX status. Returns false (and counts a drop) if the UART buffer is full.
static bool xbeeSend(uint8_t dest, const char* payload, size_t len) {
  if (len == 0 || len > 200) return false;
  uint8_t fid = nextFrameId;
  nextFrameId = (nextFrameId >= 255) ? 1 : nextFrameId + 1;

  uint8_t pkt[4 + 14 + 200 + 1];
  uint16_t frameLen = (uint16_t)(14 + len);
  pkt[0] = 0x7E;
  pkt[1] = (uint8_t)(frameLen >> 8);
  pkt[2] = (uint8_t)(frameLen & 0xFF);
  pkt[3] = 0x10;                        // Transmit Request
  pkt[4] = fid;                         // frame id
  memcpy(&pkt[5], dest == 0 ? XBEE_MAC_RX1 : XBEE_MAC_RX2, 8);
  pkt[13] = 0xFF; pkt[14] = 0xFE;       // 16 bit address unknown
  pkt[15] = 0x00;                       // broadcast radius
  pkt[16] = 0x00;                       // options
  memcpy(&pkt[17], payload, len);
  uint32_t sum = 0;
  for (size_t i = 3; i < 17 + len; i++) sum += pkt[i];
  pkt[17 + len] = (uint8_t)(0xFF - (sum & 0xFF));
  size_t total = 18 + len;

  if ((size_t)XBEE_SERIAL.availableForWrite() < total) { txDropped++; return false; }
  XBEE_SERIAL.write(pkt, total);
  frameIdDest[fid] = dest;
  txSentX++;
  sentSinceStatus++;
  return true;
}

static void noteTxStatus(uint8_t id, uint8_t status) {
  uint8_t dest = frameIdDest[id] & 1;
  lastStatusCode = status;
  lastStatusMs = millis();
  sentSinceStatus = 0;
  if (status == 0) statusOk[dest]++; else statusFail[dest]++;
}

static bool inFlightNow() {
  return fsmState >= FSM_ASCENT && fsmState <= FSM_MAIN;
}

// payload is the RF data of a received frame. Accepts "A" or "B" (either case),
// optionally wrapped in CR / LF / space / NUL (XCTU adds a CR in some modes).
static void handleCommand(const uint8_t* p, size_t n) {
  // PING → reply with PONG via XBee (to RX1)
  if ((n >= 4 && memcmp(p, "PING",  4) == 0) ||
      (n >= 5 && memcmp(p, "$PING", 5) == 0)) {
    char body[80], out[96];
    snprintf(body, sizeof(body), "%s,PONG,%lu,%u,%.2f",
             TEAM_ID, (unsigned long)packetCount, (unsigned)fsmState, (double)vbat);
    snprintf(out, sizeof(out), "$%s*%02X\r\n", body, (unsigned)xorChecksum(body));
    xbeeSend(0, out, strlen(out));
    beepOnce(80);
    cmdAccepted++; return;
  }

  size_t a = 0, b = n;
  while (a < b && (p[a] == '\r' || p[a] == '\n' || p[a] == ' ' || p[a] == 0)) a++;
  while (b > a && (p[b - 1] == '\r' || p[b - 1] == '\n' || p[b - 1] == ' ' || p[b - 1] == 0)) b--;
  if (b - a != 1) { cmdIgnored++; return; }
  char c = (char)p[a];
  if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
  uint32_t now = millis();
  if (c == 'A') {
    if (!txEnabled) {                    // restart: clear the "no status" timers
      lastStatusMs = now;
      sentSinceStatus = 0;
      beepOnce(150);
    }
    txEnabled = true;
    cmdAccepted++; lastCmd = 'A'; lastCmdMs = now;
  } else if (c == 'B') {
    if (inFlightNow() && !ALLOW_STOP_IN_FLIGHT) {
      cmdRejected++; lastCmd = 'b'; lastCmdMs = now;     // lower case = refused
      return;
    }
    if (txEnabled) beepOnce(600);
    txEnabled = false;
    cmdAccepted++; lastCmd = 'B'; lastCmdMs = now;
  } else if (c == 'E') {
    drogueFired  = true;   // blocks energizeDrogueHold() from re-firing on PAD re-entry
    flagSolenoid = true;
    beepOnce(300);
    if (inFlightNow() && launchConfirmed) {
      // Confirmed in-flight: release latch, advance FSM
      digitalWriteFast(PIN_SOL_DROGUE, LOW);
      droguePulseOffMs = 0;
      if (fsmState < FSM_DEPLOY) { tDeployMs = now; fsmState = FSM_DEPLOY; }
    } else {
      // Ground / pad / post-flight — 1 s pulse every press, FSM unchanged
      digitalWriteFast(PIN_SOL_DROGUE, LOW);
      droguePulseOffMs = now + 1000;
    }
    cmdAccepted++; lastCmd = 'E'; lastCmdMs = now;
    Serial.println("[CMD] EJECT received — solenoid fired");
  } else {
    cmdIgnored++;
  }
}

// Incoming API frames: we need TX status (0x89 / 0x8B) and received data
// (0x80 64 bit source, 0x81 16 bit source, 0x90 receive packet) for commands.
// Any other frame is parsed for framing and ignored.
static void xbeePoll() {
  static uint8_t st = 0;                // 0 idle, 1 len hi, 2 len lo, 3 data, 4 checksum
  static uint16_t flen = 0, fpos = 0;
  static uint8_t fdata[40];
  static uint8_t fsum = 0;
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
          // 802.15.4 TX status: type, frame id, status
          noteTxStatus(fdata[1], fdata[2]);
        } else if (fpos >= 6 && fdata[0] == 0x8B) {
          // extended TX status: type, id, addr16(2), retries, delivery, discovery
          noteTxStatus(fdata[1], fdata[5]);
        } else if (fpos == flen && flen <= 24) {
          // received RF data: payload offset depends on the frame type
          size_t off = 0;
          if (fdata[0] == 0x80) off = 11;        // type, addr64(8), rssi, options
          else if (fdata[0] == 0x81) off = 5;    // type, addr16(2), rssi, options
          else if (fdata[0] == 0x90) off = 12;   // type, addr64(8), addr16(2), options
          if (off > 0 && flen > off) handleCommand(&fdata[off], flen - off);
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
// 11. LORA (REYAX RYLR896) CONFIG + BEACON
// ============================================================================
static void loraPoll() {
  static char line[96];
  static uint8_t idx = 0;
  int guard = 128;
  while (guard-- > 0 && LORA_SERIAL.available()) {
    char c = (char)LORA_SERIAL.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[idx] = '\0';
      if (idx > 0) {
        if (loraCfgWaiting) {
          if (strncmp(line, "+OK", 3) == 0) { loraCfgWaiting = false; loraCfgStep++; }
          else if (strncmp(line, "+ERR", 4) == 0) { loraCfgWaiting = false; loraCfgStep++; loraCfgErrors++; }
        }
        if (strncmp(line, "+RCV", 4) == 0 && strstr(line, "PING") != nullptr) {
          beepOnce(80);
          LORA_SERIAL.print("AT+SEND=0,4,PONG\r\n");
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
    case 3: snprintf(cmd, sizeof(cmd), "AT+ADDRESS=%u", LORA_ADDRESS); break;
    case 4: snprintf(cmd, sizeof(cmd), "AT+CRFOP=%u", LORA_POWER_DBM); break;
    default: loraReady = true; return;
  }
  LORA_SERIAL.print(cmd);
  LORA_SERIAL.print("\r\n");
  loraCfgSentMs = now;
  loraCfgWaiting = true;
}

static void loraSendBeacon(uint32_t count) {
  if (!loraReady) return;
  char beacon[100];
  size_t n = buildBeacon(beacon, sizeof(beacon), count);
  if (n == 0) return;
  char cmd[140];
  int m = snprintf(cmd, sizeof(cmd), "AT+SEND=0,%u,%s\r\n", (unsigned)n, beacon);
  if (m <= 0 || (size_t)m >= sizeof(cmd)) return;
  if (LORA_SERIAL.availableForWrite() < m) { loraDropped++; return; }
  LORA_SERIAL.write((const uint8_t*)cmd, (size_t)m);
  loraSent++;
}

// ============================================================================
// 12. INDICATORS
// ============================================================================
static void updateIndicators() {
  uint32_t now = millis();
  bool sensorFault = (now - baroLastGoodMs > 500) || (now - imuLastGoodMs > 500);
  bool r = false, g = false, b = false, buz = false;
  if (sensorFault && calPhase == 2) {
    r = ((now / 125) & 1);                       // fast red blink
  } else if (fsmState == FSM_BOOT || fsmState == FSM_TEST) {
    r = true;
  } else if (fsmState == FSM_PAD) {
    g = true;
  } else if (fsmState == FSM_IMPACT) {
    g = ((now / 500) & 1);
    buz = (now % 3000) < 800;                    // recovery locator beeps
  } else {
    b = true;
  }
  if (!txEnabled) {                              // stopped by command: red/blue alternate
    r = ((now / 500) & 1); g = false; b = !r;
  }
  if ((int32_t)(beepUntilMs - now) > 0) buz = true;
  digitalWrite(PIN_LED_R, r);
  digitalWrite(PIN_LED_G, g);
  digitalWrite(PIN_LED_B, b);
  digitalWrite(PIN_BUZZER, buz);
}

#if HAVE_WDT
static void wdtWarning() { if (Serial) Serial.println("WDT WARNING: loop stalled"); }
#endif

// ============================================================================
// 13. SETUP / LOOP
// ============================================================================
void setup() {
  pinMode(PIN_BUZZER,     OUTPUT);
  pinMode(PIN_LED_R,      OUTPUT);
  pinMode(PIN_LED_G,      OUTPUT);
  pinMode(PIN_LED_B,      OUTPUT);
  pinMode(PIN_SOL_DROGUE, OUTPUT);
  pinMode(PIN_SOL_MAIN,   OUTPUT);
  digitalWriteFast(PIN_SOL_DROGUE, HIGH);  // hold latch closed until ejection
  digitalWriteFast(PIN_SOL_MAIN,   HIGH);  // hold latch closed until main deploy
  digitalWrite(PIN_BUZZER, LOW);
  digitalWrite(PIN_LED_R, HIGH);

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
  Wire.setTimeout(50);   // 50 ms hard cap — a stuck sensor can no longer stall the loop
  initSensors();
  initSdCard();

  resumeCandidate = recordLoad() && savedRec.inFlight == 1;
  calPhase0Start = millis();
  memset(frameIdDest, 0, sizeof(frameIdDest));

  uint32_t now = millis();
  nextSensorMs = nextTxMs = now + 20;
  nextLoraMs = now + LORA_PERIOD_MS;
  nextStatusMs = now + STATUS_PERIOD_MS;
  nextBattMs = now;
  nextSdFlushMs = now + SD_FLUSH_PERIOD_MS;
  lastStatusMs = now;
  baroLastGoodMs = imuLastGoodMs = now;

#if HAVE_WDT
  WDT_timings_t cfg;
  cfg.trigger = 1;       // seconds: warning callback
  cfg.timeout = 2;       // seconds: hardware reset
  cfg.callback = wdtWarning;
  wdt.begin(cfg);
#else
  if (Serial) Serial.println("WARNING: Watchdog_t4 library not installed, NO watchdog");
#endif
}

void loop() {
  feedWatchdog();
  uint32_t now = millis();

  processGnss();
  xbeePoll();
  serviceActuators();
  loraPoll();
  loraConfigure();
  xbeeWatch();

  if ((int32_t)(now - nextSensorMs) >= 0) {
    nextSensorMs += SENSOR_PERIOD_MS;
    if ((int32_t)(now - nextSensorMs) >= 0) nextSensorMs = now + SENSOR_PERIOD_MS;   // after a stall skip ahead, no burst
    sensorTick();
    if (sdActive) {
      char csv[140];
      snprintf(csv, sizeof(csv), "%lu,%lu,%u,%.2f,%.2f,%.1f,%.2f,%.2f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.6f,%.6f,%u,%u,%u\n",
               (unsigned long)now, (unsigned long)packetCount, (unsigned)fsmState,
               (double)aglRaw, (double)(baroPa / 100.0f), (double)baroTempC,
               (double)vbat, (double)vbatSol,
               (double)accel[0], (double)accel[1], (double)accel[2],
               (double)gyro[0], (double)gyro[1], (double)gyro[2],
               gLat, gLon, (unsigned)gSats,
               drogueFired ? 1u : 0u, mainFired ? 1u : 0u);
      logToSdQueue(csv);
    }
  }

  if ((int32_t)(now - nextBattMs) >= 0) {
    nextBattMs += 100;
    if ((int32_t)(now - nextBattMs) >= 0) nextBattMs = now + 100;
    readBattery();
  }

  if ((int32_t)(now - nextSdFlushMs) >= 0) {
    nextSdFlushMs += SD_FLUSH_PERIOD_MS;
    if ((int32_t)(now - nextSdFlushMs) >= 0) nextSdFlushMs = now + SD_FLUSH_PERIOD_MS;
    flushSdQueue();
  }

  if ((int32_t)(now - nextTxMs) >= 0) {
    nextTxMs += TX_PERIOD_MS;
    if ((int32_t)(now - nextTxMs) >= 0) nextTxMs = now + TX_PERIOD_MS;   // after a stall skip ahead, no burst
    // While stopped nothing is sent and the counter does not advance, so the
    // GCS sees a contiguous PACKET_COUNT again after 'A'.
    size_t n = 0;
    char frame[256];
    if (txEnabled) {
      packetCount++;
      n = buildTelemetry(frame, sizeof(frame) - 3, packetCount, now);
    }
    if (n > 0) {
      frame[n++] = '\r';
      frame[n++] = '\n';
      frame[n] = '\0';
      xbeeSend((packetCount & 1) ? 0 : 1, frame, n);   // odd -> RX1, even -> RX2
    }
  }

  if ((int32_t)(now - nextLoraMs) >= 0) {
    nextLoraMs += LORA_PERIOD_MS;
    if ((int32_t)(now - nextLoraMs) >= 0) nextLoraMs = now + LORA_PERIOD_MS;   // after a stall skip ahead, no burst
    loraSendBeacon(packetCount);
  }

  updateIndicators();

  if ((int32_t)(now - nextStatusMs) >= 0) {
    nextStatusMs += STATUS_PERIOD_MS;
    if ((int32_t)(now - nextStatusMs) >= 0) nextStatusMs = now + STATUS_PERIOD_MS;   // after a stall skip ahead, no burst
    if (Serial) {
      Serial.printf("[STAT] t=%lus fsm=%u agl=%.1fm P=%.1fhPa T=%.1fC V=%.2f sats=%u fix=%u | "
                    "tx=%lu drop=%lu okRX1=%lu okRX2=%lu failRX1=%lu failRX2=%lu st=0x%02X | "
                    "lora=%lu ready=%u cfgErr=%u | i2cRec=%lu imuF=%lu baroF=%lu xbRec=%lu "
                    "nmea=%lu/%lu id=%u%u wdt=%d | txOn=%u cmd=%c ok=%lu refused=%lu ignored=%lu\n",
                    (unsigned long)(now / 1000UL), (unsigned)fsmState, (double)aglRaw,
                    (double)(baroPa / 100.0f), (double)baroTempC, (double)vbat,
                    (unsigned)gSats, (unsigned)gnssFixValid(),
                    (unsigned long)txSentX, (unsigned long)txDropped,
                    (unsigned long)statusOk[0], (unsigned long)statusOk[1],
                    (unsigned long)statusFail[0], (unsigned long)statusFail[1], lastStatusCode,
                    (unsigned long)loraSent, (unsigned)loraReady, (unsigned)loraCfgErrors,
                    (unsigned long)i2cRecoveries, (unsigned long)imuFailures,
                    (unsigned long)baroFailures, (unsigned long)xbeeRecoveries,
                    (unsigned long)nmeaGood, (unsigned long)nmeaBad,
                    (unsigned)imuIdOk, (unsigned)baroIdOk, (int)HAVE_WDT,
                    (unsigned)txEnabled, lastCmd, (unsigned long)cmdAccepted,
                    (unsigned long)cmdRejected, (unsigned long)cmdIgnored);
    }
  }
}
