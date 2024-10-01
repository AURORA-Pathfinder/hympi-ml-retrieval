#!/bin/bash

module load python/GEOSpyD/Min23.5.2-0_py3.11

echo "Starting processing at " $(date)
free -h > free_report.txt
#rm -rf done.csv
python makeday.py
echo "Ending processing at " $(date)

scancel 34798381

