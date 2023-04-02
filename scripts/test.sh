#!/bin/bash
gcc test/serial_multiplication.c -o test/serial.x $1
./test/serial.x
rm ./test/serial.x
