#!/bin/bash
gcc test/serial_multiplication.c -o test/serial.x
./test/serial.x
rm ./test/serial.x
