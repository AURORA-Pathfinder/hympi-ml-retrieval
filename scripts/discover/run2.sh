#!/bin/bash

module load python/GEOSpyD/Min23.5.2-0_py3.11

echo "Starting processing at " $(date)
free -h > free_report.txt
python gen_new_redo.py
python data_process.py
echo "Ending processing at " $(date)


