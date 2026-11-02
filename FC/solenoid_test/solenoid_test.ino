// Solenoid hardware test — pins 22 and 23 held HIGH permanently.
// Upload this to verify the D4184 modules and solenoid wiring
// before using the full FC firmware.

void setup() {
  pinMode(22, OUTPUT);
  pinMode(23, OUTPUT);
  digitalWriteFast(22, HIGH);
  digitalWriteFast(23, LOW);
}

void loop() {}
