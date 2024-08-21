#!/bin/bash


for i in `seq 0 10`
do
    python main.py --config-name HSEL
done

for i in `seq 0 10`
do
    python main.py --config-name HSEL_BSL
done

for i in `seq 0 10`
do
    python main.py --config-name ATMS
done

for i in `seq 0 10`
do
    python main.py --config-name ATMS_BSL
done

for i in `seq 0 10`
do
    python main.py --config-name ATMS_HA
done
