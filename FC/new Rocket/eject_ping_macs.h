#pragma once
#include <stdint.h>

// Ejection ping destination MACs — XBee 64-bit addresses.
// FC sends a dedicated ejection-event ping to both of these
// after the ejection solenoid fires.
const uint8_t XBEE_EJECT_RX1[8] =
{
  0x00, 0x13, 0xA2, 0x00,
  0x42, 0xA0, 0x95, 0xF5
};

const uint8_t XBEE_EJECT_RX2[8] =
{
  0x00, 0x13, 0xA2, 0x00,
  0x42, 0xA0, 0x96, 0x41
};
