#!/bin/bash

module load python/GEOSpyD/Min23.5.2-0_py3.11

echo "Starting processing at " $(date)
python data_ingest_all_v4.py 46
echo "Ending processing at " $(date)


