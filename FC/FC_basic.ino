/*
  FC_basic.ino — Teensy 4.1 Basic Flight Computer
  ================================================
  Hardware:
    - Teensy 4.1 (600 MHz Cortex-M7)
    - XBee 3 Pro (XB3-24AST) 802.15.4 API-1 on Serial2 (pins 7/8)
    - GNSS: 7Semi L89HA on Serial7 (9600 baud)
    - IMU: LSM6DSO32 via I2C (SDA=18, SCL=19)
    - Baro: BMP581 via I2C
    - Drogue solenoid: Pin 22 (D4184 MOSFET)
    - Main solenoid:   Pin 23 (D4184 MOSFET)
    - Buzzer: Pin 4
    - RGB LED: R=30, G=31, B=32
    - Battery ADC: A10 (FC 2S), A11 (Solenoid 3S)

  GCS UPLINK COMMANDS (single ASCII byte via XBee):
    'A' — Start telemetry
    'B' — Stop  telemetry (blocked mid-flight)
    'E' — Fire drogue solenoid (pin 22) unconditionally
    "PING" — Link test; replies with PONG
*/

#include <Arduino.h>
#include <Wire.h>
#include <SD.h>
#include <math.h>

// ─── Identity ───────────────────────────────────────────────────────────────
static const char TEAM_ID[] = "2025ASI001";

// ─── Pins ───────────────────────────────────────────────────────────────────
#define XBEE_SERIAL  Serial2
#define GNSS_SERIAL  Serial7
constexpr uint8_t PIN_SOL_DROGUE = 22;
constexpr uint8_t PIN_SOL_MAIN   = 23;
constexpr uint8_t PIN_BUZZER     = 4;
constexpr uint8_t PIN_LED_R      = 30;
constexpr uint8_t PIN_LED_G      = 31;
constexpr uint8_t PIN_LED_B      = 32;
constexpr uint8_t PIN_ADC_FC     = 24;   // A10
constexpr uint8_t PIN_ADC_SOL    = 25;   // A11

// ─── Baud rates ─────────────────────────────────────────────────────────────
constexpr uint32_t XBEE_BAUD = 115200;
constexpr uint32_t GNSS_BAUD =   9600;

// ─── Timing (ms) ────────────────────────────────────────────────────────────
constexpr uint32_t SENSOR_PERIOD_MS =  10;   // 100 Hz
constexpr uint32_t TX_PERIOD_MS     = 100;   // 10 Hz
constexpr uint32_t STATUS_PERIOD_MS = 1000;  // 1 Hz USB debug

// ─── XBee broadcast address ─────────────────────────────────────────────────
static const uint8_t XBEE_BCAST[8] = {0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0xFF};

// ─── I2C sensor addresses ────────────────────────────────────────────────────
constexpr uint8_t LSM_ADDR = 0x6A;
constexpr uint8_t BMP_ADDR = 0x47;

// ─── Physical constants ──────────────────────────────────────────────────────
constexpr float G = 9.80665f;
constexpr float ACCEL_SCALE = (32.0f / 32768.0f) * G;   // ±32 g LSM6DSO32
constexpr float GYRO_SCALE  = 0.070f;                    // ±2000 dps
constexpr float BATT_SCALE  = (3.30f / 4095.0f) * 4.30f;

// ─── FSM states ─────────────────────────────────────────────────────────────
constexpr uint8_t FSM_BOOT    = 0;
constexpr uint8_t FSM_PAD     = 2;
constexpr uint8_t FSM_ASCENT  = 3;
constexpr uint8_t FSM_DEPLOY  = 4;
constexpr uint8_t FSM_DESCENT = 5;
constexpr uint8_t FSM_MAIN    = 6;
constexpr uint8_t FSM_IMPACT  = 7;

// ─── Flight thresholds ───────────────────────────────────────────────────────
constexpr float    LAUNCH_ACCEL_MS2  = 2.5f * G;  // 2.5 g
constexpr uint8_t  LAUNCH_SAMPLES   = 5;           // 50 ms
constexpr float    APOGEE_DROP_M    = 3.0f;
constexpr uint8_t  APOGEE_SAMPLES   = 6;
constexpr uint32_t BURNOUT_MS       = 3000;        // 3 s boost lockout
constexpr uint32_t DEPLOY_HOLD_MS   = 1000;
constexpr float    MAIN_ALT_M       = 400.0f;
constexpr uint32_t MAIN_PULSE_MS    = 3000;        // 3 s main solenoid pulse
constexpr float    LAND_SPEED_MS    = 0.7f;
constexpr uint8_t  LAND_WINDOWS     = 3;
constexpr float    LAND_MAX_AGL     = 60.0f;

// ─── Calibration ─────────────────────────────────────────────────────────────
constexpr uint16_t CAL_SAMPLES = 300;              // 3 s at 100 Hz

// ============================================================================
// STATE
// ============================================================================
static float accel[3] = {0, 0, G};
static float gyro[3]  = {0, 0, 0};
static float gyroBias[3] = {0, 0, 0};
static float baroPa   = 101325.0f;
static float baroTempC = 20.0f;
static float groundPa  = 101325.0f;
static float aglRaw = 0, aglFilt = 0, aglPrev = 0, vertSpeed = 0, maxAgl = 0;
static float vbatFC = 0, vbatSol = 0;

static uint8_t  fsmState    = FSM_BOOT;
static bool     drogueFired = false, mainFired = false;
static uint32_t mainOffMs   = 0;               // when to cut main solenoid
static uint32_t droguePulseOffMs = 0;          // when to cut test pulse
static uint32_t tLaunchMs = 0, tDeployMs = 0;
static uint8_t  launchCnt = 0, apogeeCnt = 0, stillCnt = 0;
static uint32_t tick100 = 0;                   // counts 100 Hz ticks for 1 Hz vertSpeed

// Calibration
static bool     calDone = false;
static uint16_t calCount = 0;
static double   calPaSum = 0;
static float    calGSum[3] = {0, 0, 0};

// Telemetry
static bool     txEnabled   = true;
static uint32_t packetCount = 0;
static uint8_t  nextFrameId = 1;

// GNSS
static double  gLat = 0, gLon = 0;
static float   gAlt = 0;
static uint8_t gSats = 0, gQuality = 0;
static char    gTime[16] = "";
static uint32_t gLastGgaMs = 0;

// Schedulers
static uint32_t nextSensorMs = 0, nextTxMs = 0, nextStatusMs = 0;
static uint32_t beepUntilMs  = 0;
static uint32_t xbeeLastRxMs = 0;   // last time any uplink frame arrived
static uint32_t xbeeWatchMs  = 0;   // last time watchdog ran

// ============================================================================
// UTILITIES
// ============================================================================
static uint8_t xorCs(const char* s) {
  uint8_t c = 0;
  while (*s) c ^= (uint8_t)*s++;
  return c;
}

static void beep(uint16_t ms) { beepUntilMs = millis() + ms; }

static float pressToAgl(float pa) {
  return 44330.0f * (1.0f - powf(pa / groundPa, 0.190294957f));
}

// ============================================================================
// I2C HELPERS
// ============================================================================
static bool i2cWrite(uint8_t dev, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(dev);
  Wire.write(reg); Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool i2cRead(uint8_t dev, uint8_t reg, uint8_t* buf, uint8_t n) {
  Wire.beginTransmission(dev);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(dev, n) != n)    return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = (uint8_t)Wire.read();
  return true;
}

static void initSensors() {
  // LSM6DSO32: 416 Hz, ±32 g accel, ±2000 dps gyro
  i2cWrite(LSM_ADDR, 0x12, 0x44); // CTRL3_C: BDU + auto-inc
  i2cWrite(LSM_ADDR, 0x10, 0x64); // CTRL1_XL: 416 Hz ±32g
  i2cWrite(LSM_ADDR, 0x11, 0x6C); // CTRL2_G:  416 Hz ±2000dps

  // BMP581: 100 Hz continuous, 8x pressure OSR
  i2cWrite(BMP_ADDR, 0x36, 0x58); // OSR_CONFIG
  i2cWrite(BMP_ADDR, 0x37, 0x2B); // ODR_CONFIG
}

// ============================================================================
// SENSOR READS
// ============================================================================
static bool readImu() {
  uint8_t b[12];
  if (!i2cRead(LSM_ADDR, 0x22, b, 12)) return false;
  for (uint8_t i = 0; i < 3; i++) {
    int16_t rg = (int16_t)((uint16_t)b[2*i]   | ((uint16_t)b[2*i+1]   << 8));
    int16_t ra = (int16_t)((uint16_t)b[6+2*i] | ((uint16_t)b[6+2*i+1] << 8));
    gyro[i]  = (float)rg * GYRO_SCALE;
    accel[i] = (float)ra * ACCEL_SCALE;
  }
  return true;
}

static bool readBaro() {
  uint8_t b[6];
  if (!i2cRead(BMP_ADDR, 0x1D, b, 6)) return false;
  int32_t t = (int32_t)(((uint32_t)b[2]<<16)|((uint32_t)b[1]<<8)|b[0]);
  if (t & 0x800000) t -= 0x1000000;
  uint32_t p = ((uint32_t)b[5]<<16)|((uint32_t)b[4]<<8)|b[3];
  baroTempC = (float)t / 65536.0f;
  baroPa    = (float)p / 64.0f;
  return baroPa > 30000.0f && baroPa < 110000.0f;
}

static void readBatt() {
  vbatFC  = analogRead(PIN_ADC_FC)  * BATT_SCALE;
  vbatSol = analogRead(PIN_ADC_SOL) * BATT_SCALE;
}

// ============================================================================
// GNSS
// ============================================================================
static int hexV(char c) {
  if (c>='0'&&c<='9') return c-'0';
  if (c>='A'&&c<='F') return c-'A'+10;
  if (c>='a'&&c<='f') return c-'a'+10;
  return -1;
}

static double nmeaDeg(const char* raw, char hem, bool isLat) {
  double v = atof(raw);
  int d = (int)(v/100); double m = v - d*100;
  double deg = d + m/60.0;
  return (hem=='S'||hem=='W') ? -deg : deg;
}

static void parseNmea(char* s) {
  if (s[0]!='$') return;
  char* star = strchr(s,'*');
  if (!star||!star[1]||!star[2]) return;
  uint8_t cs = 0;
  for (char* p=s+1; p<star; p++) cs ^= (uint8_t)*p;
  if (cs != (uint8_t)((hexV(star[1])<<4)|hexV(star[2]))) return;
  *star = '\0';

  char* t[16]; uint8_t c=0;
  t[c++] = s+1;
  for (char* p=s+1; *p&&c<16; p++) if (*p==','){*p='\0';t[c++]=p+1;}
  if (c<10 || strcmp(t[0]+2,"GGA")!=0) return;

  gQuality = (uint8_t)atoi(t[6]);
  gSats    = (uint8_t)atoi(t[7]);
  gLastGgaMs = millis();
  if (gQuality==0) return;
  gLat = nmeaDeg(t[2], t[3][0], true);
  gLon = nmeaDeg(t[4], t[5][0], false);
  float ga = (float)atof(t[9]);
  if (ga>-500&&ga<9000) gAlt = ga;
  strncpy(gTime, t[1], sizeof(gTime)-1);
}

static void processGnss() {
  static char buf[128]; static uint8_t idx=0;
  while (GNSS_SERIAL.available()) {
    char c = (char)GNSS_SERIAL.read();
    if (c=='\r') continue;
    if (c=='\n') { if(idx>6){buf[idx]='\0';parseNmea(buf);} idx=0; }
    else if (idx<sizeof(buf)-1) buf[idx++]=c;
    else idx=0;
  }
}

static bool gpsFix() { return gQuality>0 && (millis()-gLastGgaMs)<3000; }

// ============================================================================
// SOLENOID ACTUATORS
// ============================================================================
static void serviceActuators() {
  uint32_t now = millis();
  if (mainOffMs && (int32_t)(now-mainOffMs)>=0) {
    digitalWriteFast(PIN_SOL_MAIN, LOW);
    mainOffMs = 0;
    Serial.println("[ACT] Main solenoid OFF");
  }
  if (droguePulseOffMs && (int32_t)(now-droguePulseOffMs)>=0) {
    digitalWriteFast(PIN_SOL_DROGUE, LOW);
    droguePulseOffMs = 0;
    Serial.println("[ACT] Drogue test pulse OFF");
  }
}

static void releaseDrogue() {
  if (drogueFired) return;
  digitalWriteFast(PIN_SOL_DROGUE, LOW);
  drogueFired = true;
  beep(300);
  Serial.println("[ACT] Drogue released");
}

static void fireMain() {
  if (mainFired) return;
  digitalWriteFast(PIN_SOL_MAIN, HIGH);
  mainOffMs = millis() + MAIN_PULSE_MS;
  mainFired = true;
  beep(500);
  Serial.println("[ACT] Main fired — 3 s pulse");
}

// ============================================================================
// CALIBRATION & FLIGHT FSM
// ============================================================================
static void sensorTick() {
  bool imuOk  = readImu();
  bool baroOk = readBaro();
  uint32_t now = millis();

  // ── Calibration phase: collect 3 s of stillness ──────────────────────────
  if (!calDone) {
    if (!imuOk || !baroOk) return;
    float an = sqrtf(accel[0]*accel[0]+accel[1]*accel[1]+accel[2]*accel[2]);
    bool still = fabsf(gyro[0])<10 && fabsf(gyro[1])<10 && fabsf(gyro[2])<10
                 && an > 0.7f*G && an < 1.3f*G;
    if (!still) { calCount=0; calPaSum=0; calGSum[0]=calGSum[1]=calGSum[2]=0; return; }
    for (uint8_t i=0;i<3;i++) calGSum[i] += gyro[i];
    calPaSum += baroPa;
    if (++calCount >= CAL_SAMPLES) {
      for (uint8_t i=0;i<3;i++) gyroBias[i] = calGSum[i]/CAL_SAMPLES;
      groundPa = (float)(calPaSum/CAL_SAMPLES);
      calDone  = true;
      aglFilt  = aglPrev = maxAgl = 0;
      // Energize drogue solenoid HIGH — holds latch closed on pad
      digitalWriteFast(PIN_SOL_DROGUE, HIGH);
      fsmState = FSM_PAD;
      beep(250);
      Serial.printf("[CAL] Done: groundPa=%.1f\n", groundPa);
    }
    return;
  }

  // ── Apply gyro bias ───────────────────────────────────────────────────────
  for (uint8_t i=0;i<3;i++) gyro[i] -= gyroBias[i];

  // ── Altitude filter ───────────────────────────────────────────────────────
  if (baroOk) {
    aglRaw  = pressToAgl(baroPa);
    aglFilt += 0.15f * (aglRaw - aglFilt);
  }

  // ── 1 Hz vertical speed ───────────────────────────────────────────────────
  if (++tick100 >= 100) {
    tick100   = 0;
    vertSpeed = aglFilt - aglPrev;
    aglPrev   = aglFilt;
    if (fabsf(vertSpeed) < LAND_SPEED_MS) { if(stillCnt<255) stillCnt++; }
    else stillCnt = 0;
  }

  float an = sqrtf(accel[0]*accel[0]+accel[1]*accel[1]+accel[2]*accel[2]);

  // ── FSM ───────────────────────────────────────────────────────────────────
  switch (fsmState) {

    case FSM_PAD:
      if (imuOk && an > LAUNCH_ACCEL_MS2) launchCnt++; else launchCnt=0;
      if (launchCnt >= LAUNCH_SAMPLES) {
        tLaunchMs = now; maxAgl = aglFilt;
        apogeeCnt = stillCnt = launchCnt = 0;
        fsmState  = FSM_ASCENT;
        Serial.println("[FSM] LAUNCH");
      }
      break;

    case FSM_ASCENT:
      if (aglFilt > maxAgl) maxAgl = aglFilt;
      if ((now-tLaunchMs > BURNOUT_MS) && (maxAgl-aglFilt) > APOGEE_DROP_M) {
        if (++apogeeCnt >= APOGEE_SAMPLES) {
          tDeployMs = now;
          releaseDrogue();
          fsmState  = FSM_DEPLOY;
          Serial.printf("[FSM] APOGEE %.1fm — drogue released\n", maxAgl);
        }
      } else { apogeeCnt = 0; }
      break;

    case FSM_DEPLOY:
      if (now-tDeployMs >= DEPLOY_HOLD_MS) fsmState = FSM_DESCENT;
      break;

    case FSM_DESCENT:
      if (aglFilt <= MAIN_ALT_M && vertSpeed < -1.0f && drogueFired) {
        fireMain();
        fsmState = FSM_MAIN;
        Serial.printf("[FSM] MAIN at %.1fm\n", aglFilt);
      }
      break;

    default: break;
  }

  // ── Landing check (any post-apogee state) ────────────────────────────────
  if (fsmState>=FSM_DEPLOY && fsmState<=FSM_MAIN
      && stillCnt>=LAND_WINDOWS && aglFilt<LAND_MAX_AGL) {
    digitalWriteFast(PIN_SOL_DROGUE, LOW);
    digitalWriteFast(PIN_SOL_MAIN,   LOW);
    fsmState = FSM_IMPACT;
    Serial.println("[FSM] IMPACT — solenoids off");
  }
}

// ============================================================================
// TELEMETRY FRAME
// ============================================================================
static size_t buildFrame(char* out, size_t cap) {
  bool fix = gpsFix();
  char lat[18]="", lon[18]="", nalt[12]="";
  if (fix) {
    snprintf(lat,  sizeof(lat),  "%.6f", gLat);
    snprintf(lon,  sizeof(lon),  "%.6f", gLon);
    snprintf(nalt, sizeof(nalt), "%.1f", (double)gAlt);
  }
  char body[240];
  snprintf(body, sizeof(body),
    "%s,ROCKET,%lu.%02lu,%lu,%.2f,%.2f,%.2f,%.2f,%s,%s,%s,%s,%u,"
    "%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%u,%u,%u",
    TEAM_ID,
    (unsigned long)(millis()/1000),(unsigned long)((millis()%1000)/10),
    (unsigned long)packetCount,
    (double)aglRaw,(double)(baroPa/100.0f),(double)baroTempC,(double)vbatFC,
    gTime[0] ? gTime : "", lat, lon, nalt, (unsigned)gSats,
    (double)accel[0],(double)accel[1],(double)accel[2],
    (double)gyro[0], (double)gyro[1], (double)gyro[2],
    (unsigned)fsmState, drogueFired?1u:0u, mainFired?1u:0u);
  return (size_t)snprintf(out, cap, "$%s*%02X\r\n", body, (unsigned)xorCs(body));
}

// ============================================================================
// XBEE API-1 SEND
// ============================================================================
static bool xbeeSend(const uint8_t* dest, const char* payload, size_t n) {
  if (n==0||n>220) return false;
  uint8_t fid = nextFrameId;
  nextFrameId = (nextFrameId>=255) ? 1 : nextFrameId+1;

  uint16_t flen = (uint16_t)(14+n);
  uint8_t  pkt[4+14+220+1];
  pkt[0]=0x7E; pkt[1]=(uint8_t)(flen>>8); pkt[2]=(uint8_t)(flen&0xFF);
  pkt[3]=0x10; pkt[4]=fid;
  memcpy(&pkt[5], dest, 8);
  pkt[13]=0xFF; pkt[14]=0xFE; pkt[15]=0x00; pkt[16]=0x00;
  memcpy(&pkt[17], payload, n);
  uint32_t s=0; for(size_t i=3;i<17+n;i++) s+=pkt[i];
  pkt[17+n] = (uint8_t)(0xFF-(s&0xFF));
  XBEE_SERIAL.write(pkt, 18+n);
  return true;
}

// ============================================================================
// UPLINK COMMAND HANDLER
// ============================================================================
// XBee watchdog: if the module locks up (RF interference, buffer overflow),
// restart the serial after 10 s of uplink silence. Uses end()/begin() rather
// than AT+FR so it works even when the module stopped accepting AT commands.
static void xbeeWatch() {
  uint32_t now = millis();
  if (now - xbeeWatchMs < 5000) return;          // check every 5 s
  xbeeWatchMs = now;
  if (xbeeLastRxMs != 0 && (now - xbeeLastRxMs) > 10000) {
    XBEE_SERIAL.end();
    delay(5);
    XBEE_SERIAL.begin(XBEE_BAUD);
    xbeeLastRxMs = now;                           // reset so we don't re-trigger immediately
    Serial.println("[XBee] watchdog restart (no uplink 10 s)");
  }
}

static void handleCommand(const uint8_t* p, size_t n, const uint8_t* senderMac) {
  if (n==0) return;
  xbeeLastRxMs = millis();    // any uplink resets the watchdog

  // PING
  if ((n>=4&&memcmp(p,"PING",4)==0)||(n>=5&&memcmp(p,"$PING",5)==0)) {
    if (fsmState<=FSM_PAD) beep(80);
    char pong[80], pkt[96];
    snprintf(pong, sizeof(pong), "%s,PONG,%lu,%u,%.2f",
             TEAM_ID,(unsigned long)packetCount,(unsigned)fsmState,(double)vbatFC);
    snprintf(pkt,  sizeof(pkt),  "$%s*%02X\r\n", pong, (unsigned)xorCs(pong));
    xbeeSend(senderMac ? senderMac : XBEE_BCAST, pkt, strlen(pkt));
    Serial.println("[CMD] PING → PONG");
    return;
  }

  // Strip whitespace, require exactly 1 byte
  size_t a=0, b=n;
  while (a<b && (p[a]<33||p[a]==0)) a++;
  while (b>a && (p[b-1]<33||p[b-1]==0)) b--;
  if (b-a!=1) return;

  char c = (char)p[a];
  if (c>='a'&&c<='z') c = c-'a'+'A';
  uint32_t now = millis();

  if (c=='A') {
    txEnabled = true;
    beep(150);
    Serial.println("[CMD] TX START");

  } else if (c=='B') {
    bool inFlight = fsmState>=FSM_ASCENT && fsmState<=FSM_MAIN;
    if (!inFlight) { txEnabled=false; beep(600); Serial.println("[CMD] TX STOP"); }
    else            Serial.println("[CMD] TX STOP refused — mid-flight");

  } else if (c=='E') {
    // Unconditional drogue eject — fires every click.
    // On ground: 150 ms HIGH pulse (audible/tactile test).
    // In flight: pin LOW immediately (never re-engage latch mid-flight).
    drogueFired = true;
    beep(300);
    if (fsmState<=FSM_PAD) {
      digitalWriteFast(PIN_SOL_DROGUE, HIGH);
      droguePulseOffMs = now + 150;
    } else {
      digitalWriteFast(PIN_SOL_DROGUE, LOW);
      droguePulseOffMs = 0;
    }
    if (fsmState<FSM_DEPLOY) { tDeployMs=now; fsmState=FSM_DEPLOY; }
    Serial.printf("[CMD] EJECT (state=%u)\n",(unsigned)fsmState);
  }
}

// ============================================================================
// XBEE RECEIVE PARSER
// ============================================================================
static void xbeePoll() {
  static uint8_t  st=0;
  static uint16_t flen=0, fpos=0;
  static uint8_t  fdata[128], fsum=0;

  while (XBEE_SERIAL.available()) {
    uint8_t b=(uint8_t)XBEE_SERIAL.read();
    switch(st) {
      case 0: if(b==0x7E) st=1; break;
      case 1: flen=(uint16_t)(b<<8); st=2; break;
      case 2:
        flen=(uint16_t)(flen|b);
        if(flen==0||flen>127){st=(b==0x7E)?1:0; break;}
        fpos=0; fsum=0; st=3; break;
      case 3:
        if(fpos<sizeof(fdata)) fdata[fpos]=b;
        fsum+=b; fpos++;
        if(fpos>=flen) st=4;
        break;
      case 4:
        st=0;
        if((uint8_t)(fsum+b)!=0xFF) break;       // bad checksum
        if(fdata[0]==0x90 && flen>=13)            // ZB receive
          handleCommand(&fdata[12], flen-12, &fdata[1]);
        else if(fdata[0]==0x80 && flen>=12)       // 64-bit receive
          handleCommand(&fdata[11], flen-11, &fdata[1]);
        else if(fdata[0]==0x81 && flen>=6)        // 16-bit receive
          handleCommand(&fdata[5],  flen-5,  nullptr);
        break;
    }
  }
}

// ============================================================================
// LED & BUZZER
// ============================================================================
static void updateIndicators() {
  uint32_t now = millis();
  bool buz = (int32_t)(beepUntilMs-now) > 0;
  bool r=false, g=false, bl=false;

  switch(fsmState) {
    case FSM_BOOT:    r=((now/250)&1); g=r; break;  // yellow blink = calibrating
    case FSM_PAD:     g=true; buz|=((now%5000)<50); break;  // green + heartbeat beep
    case FSM_ASCENT:  bl=true; break;
    case FSM_DEPLOY:  r=true; bl=true; break;        // purple
    case FSM_DESCENT: g=true; bl=true; break;        // cyan
    case FSM_MAIN:    r=true; g=true; bl=true; break;// white
    case FSM_IMPACT:  g=((now/500)&1); buz|=((now%3000)<800); break;// recovery
    default:          r=true; break;
  }
  if (!txEnabled) { r=((now/500)&1); g=false; bl=!r; }

  digitalWriteFast(PIN_LED_R, r);
  digitalWriteFast(PIN_LED_G, g);
  digitalWriteFast(PIN_LED_B, bl);
  digitalWriteFast(PIN_BUZZER, buz);
}

// ============================================================================
// SETUP & LOOP
// ============================================================================
void setup() {
  // Pin init — solenoids LOW (safe) before anything else
  pinMode(PIN_SOL_DROGUE, OUTPUT); digitalWriteFast(PIN_SOL_DROGUE, LOW);
  pinMode(PIN_SOL_MAIN,   OUTPUT); digitalWriteFast(PIN_SOL_MAIN,   LOW);
  pinMode(PIN_BUZZER,     OUTPUT); digitalWriteFast(PIN_BUZZER,     LOW);
  pinMode(PIN_LED_R, OUTPUT); pinMode(PIN_LED_G, OUTPUT); pinMode(PIN_LED_B, OUTPUT);
  digitalWriteFast(PIN_LED_R, HIGH);

  Serial.begin(115200);
  XBEE_SERIAL.begin(XBEE_BAUD);
  GNSS_SERIAL.begin(GNSS_BAUD);
  analogReadResolution(12);

  Wire.begin();
  Wire.setClock(400000);
  Wire.setTimeout(50);    // 50 ms hard cap — a stuck sensor can no longer freeze the loop
  initSensors();

  uint32_t now = millis();
  nextSensorMs = nextTxMs = now + 20;
  nextStatusMs = now + STATUS_PERIOD_MS;

  Serial.println("[BOOT] 2025ASI001 basic FC ready");
}

void loop() {
  uint32_t now = millis();

  processGnss();
  xbeePoll();
  xbeeWatch();
  serviceActuators();

  // 100 Hz sensor + FSM
  if ((int32_t)(now-nextSensorMs) >= 0) {
    nextSensorMs += SENSOR_PERIOD_MS;
    if ((int32_t)(now-nextSensorMs) >= 0) nextSensorMs = now + SENSOR_PERIOD_MS;
    sensorTick();
    readBatt();
  }

  // 20 Hz XBee downlink
  if ((int32_t)(now-nextTxMs) >= 0) {
    nextTxMs += TX_PERIOD_MS;
    if ((int32_t)(now-nextTxMs) >= 0) nextTxMs = now + TX_PERIOD_MS;
    if (txEnabled && calDone) {
      packetCount++;
      char frame[260];
      size_t n = buildFrame(frame, sizeof(frame));
      if (n > 0) xbeeSend(XBEE_BCAST, frame, n);
    }
  }

  updateIndicators();

  // 1 Hz USB diagnostic
  if ((int32_t)(now-nextStatusMs) >= 0) {
    nextStatusMs += STATUS_PERIOD_MS;
    if ((int32_t)(now-nextStatusMs) >= 0) nextStatusMs = now + STATUS_PERIOD_MS;
    Serial.printf("[STAT] t=%lus fsm=%u agl=%.1fm P=%.1fhPa T=%.1fC V=%.2f sats=%u drogue=%u main=%u\n",
      (unsigned long)(now/1000),(unsigned)fsmState,
      (double)aglRaw,(double)(baroPa/100),(double)baroTempC,(double)vbatFC,
      (unsigned)gSats,(unsigned)drogueFired,(unsigned)mainFired);
  }
}
