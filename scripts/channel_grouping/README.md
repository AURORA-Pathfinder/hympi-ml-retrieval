# Microwave Channel Grouping Scripts

This directory contains Python scripts for **microwave channel grouping** in different frequency ranges.
These scripts were prepared for analysis and processing of simulated data for specific frequency bands.

**Directory:**

`/discover/nobackup/nshahrou/for_dave`

## Scripts

- **`grouping_cli.py`**  
  Handles grouping for the **50–58 GHz** and **175–191 GHz** frequency ranges for operational purposes.  

- **`grouping_50.py`**  
  Handles grouping for the **50–58 GHz** frequency range.  

- **`grouping_183.py`**  
  Handles grouping for the **175–191 GHz** frequency range.  

- **Other channels**  
  The **89 GHz** and **165 GHz** channels are kept as-is and are not modified by these scripts.

> **Note:** The scripts currently use **hardcoded simulated data paths**. Adjust the data paths if 
needed before running in a different environment.

---

## Usage

### Operational Workflow

1. Fixing **50–58 GHz** groupings
  ```bash
    python grouping_cli.py --input /explore/nobackup/projects/ilab/projects/Aurora/data/channel_grouping_experiment/new_TB_dec_50GHZ_3p9.nc --mode 50 --output-dir /explore/nobackup/projects/ilab/projects/Aurora/results/channel_grouping_experiment --test-name newtest15 --alpha 1000
  ```

2. Fixing **175–191 GHz** groupings
  ```bash
    python grouping_cli.py --input /explore/nobackup/projects/ilab/projects/Aurora/data/channel_grouping_experiment/new_TB_dec_183GHZ_3p9.nc --mode 183 --output-dir /explore/nobackup/projects/ilab/projects/Aurora/results/channel_grouping_experiment --test-name newtest15 --alpha 1000
  ```

### Development Workflow

1. Navigate to the script directory:
   ```bash
   cd /discover/nobackup/nshahrou/for_dave
    ```

2. Run the desired grouping script:
    ```bash
    python grouping_50.py
    python grouping_183.py
    ```
