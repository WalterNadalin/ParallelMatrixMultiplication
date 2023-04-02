#!/bin/bash
gcc src/serial_multiplication.c -o src/serial.x $1
./src/serial.x
rm ./src/serial.x
