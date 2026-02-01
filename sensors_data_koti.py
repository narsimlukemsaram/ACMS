#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Project      : Agricultural Environmental Monitoring
Description  : Reads Temperature/Pressure (BMP280) and Light Intensity (BH1750)
Dependencies : smbus2, time, sys 
Platform     : Raspberry Pi 5
Developed by : Nagaraju Lakkaraju
Date         : January 2026
-------------------------------------------------------------------------------
"""

import smbus2
import time
import sys

try:
    bus = smbus2.SMBus(1)
    I2C_AVAILABLE = True
except (FileNotFoundError, PermissionError):
    print("Warning: I2C bus not found or permission denied. Running in simulation mode.")
    bus = None
    I2C_AVAILABLE = False

class MockBus:
    def read_i2c_block_data(self, addr, cmd, len):
        return [0] * len
    def write_byte_data(self, addr, reg, val):
        pass
    def close(self):
        pass

if not I2C_AVAILABLE:
    bus = MockBus()

def get_short(data, index):
    return (data[index + 1] << 8) + data[index]

def get_ushort(data, index):
    return (data[index + 1] << 8) + data[index]

# BMP280 Temperature & Pressure Sensor
# BMP280 I2C address
BMP280_I2C_ADDR = 0x76

def bmp280_get_data(i2c_bus):
    # Read calibration data (necessary for accurate readings)
    calib = i2c_bus.read_i2c_block_data(BMP280_I2C_ADDR, 0x88, 24)
    dig_T1 = get_ushort(calib, 0)
    dig_T2 = get_short(calib, 2)
    dig_T3 = get_short(calib, 4)
    dig_P1 = get_ushort(calib, 6)
    dig_P2 = get_short(calib, 8)
    dig_P3 = get_short(calib, 10)
    dig_P4 = get_short(calib, 12)
    dig_P5 = get_short(calib, 14)
    dig_P6 = get_short(calib, 16)
    dig_P7 = get_short(calib, 18)
    dig_P8 = get_short(calib, 20)
    dig_P9 = get_short(calib, 22)

    # Set mode to Normal, oversampling x1
    i2c_bus.write_byte_data(BMP280_I2C_ADDR, 0xF4, 0x27)

    # Read temperature and pressure data (6 bytes)
    data = i2c_bus.read_i2c_block_data(BMP280_I2C_ADDR, 0xF7, 6)
        
    adc_p = ((data[0] << 16) | (data[1] << 8) | data[2]) >> 4
    adc_t = ((data[3] << 16) | (data[4] << 8) | data[5]) >> 4

    # Temperature compensation math
    var1 = ((((adc_t >> 3) - (dig_T1 << 1))) * (dig_T2)) >> 11
    var2 = (((((adc_t >> 4) - (dig_T1)) * ((adc_t >> 4) - (dig_T1))) >> 12) * (dig_T3)) >> 14
    t_fine = var1 + var2
    temp_c = (t_fine * 5 + 128) >> 8
    temp_c = temp_c / 100.0

    # Pressure compensation math
    var1 = (t_fine >> 1) - 64000
    var2 = (((var1 >> 2) * (var1 >> 2)) >> 11) * dig_P6
    var2 = var2 + ((var1 * dig_P5) << 1)
    var2 = (var2 >> 2) + (dig_P4 << 16)
    var1 = (((dig_P3 * (((var1 >> 2) * (var1 >> 2)) >> 13)) >> 3) + ((dig_P2 * var1) >> 1)) >> 18
    var1 = ((32768 + var1) * dig_P1) >> 15
        
    if var1 == 0:
        pressure = 0
    else:
        pressure = (((1048576 - adc_p) - (var2 >> 12))) * 3125
        if pressure < 0x80000000:
            pressure = (pressure << 1) // var1
        else:
            pressure = (pressure // var1) * 2
        var1 = (dig_P9 * (((pressure >> 3) * (pressure >> 3)) >> 13)) >> 12
        var2 = (((pressure >> 2)) * dig_P8) >> 13
        pressure = (pressure + ((var1 + var2 + dig_P7) >> 4)) / 100.0

    #print(f"Temp: {temp_c:.2f} °C | Pressure: {pressure:.2f} hPa")
    return temp_c, pressure

# BH1750 Light Intensity Sensor
# BH1750 I2C address
BH1750_I2C_ADDR = 0x23

# Measurement modes
# High Resolution Mode (1 Lux resolution, 120ms measurement time)
CONTINUOUS_HIGH_RES_MODE = 0x10

def bh1750_get_data(i2c_bus):
    # Read 2 bytes of data from the sensor
    # The BH1750 returns [High Byte, Low Byte]
    data = bus.read_i2c_block_data(BH1750_I2C_ADDR, CONTINUOUS_HIGH_RES_MODE, 2)

    # Convert bytes to Lux
    # Formula: (High Byte << 8 | Low Byte) / 1.2
    lux = ((data[0] << 8) | data[1]) / 1.2

    # Output with flush=True so you see it immediately over SSH
    #print(f"Light Level: {lux:>8.2f} lx", end='\r', flush=True)
    return lux

def main():
    try:
        while True:
            temp_c, pressure = bmp280_get_data(bus)
            lux = bh1750_get_data(bus)

            print(f"Temperature : {temp_c:8.2f} °C")
            print(f"Pressure    : {pressure:8.2f} hPa")
            print(f"Light Level : {lux:>8.2f} lux")
            print("")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[STOP] Program stopped ...")
    except Exception as e:
        print(f"\n[Error] {e}")
    finally:
        bus.close()
        print(f"[INFO] I2C Bus Released.")

if __name__ == "__main__":
    main()

