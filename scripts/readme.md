# Data Ingest Scripts

and related detritus.

## Filelist

### Ingest Scripts

`cloud_reduce_fulldays.py` - Reads fulldays/ dataset and outputs fulldays\_reduced/ , takes desired cloud fraction as arg

`reducer.py` - The reducer reduces. Takes in fulldays\_reduced and makes fulldays\_reduced\_evenmore

`data_ingest_aero.py` - similar to the data ingest script on discover but specifically for v42 aero data

`cpl_stacker.py` - Creates the `all_cpl_DAY.npz` files. The spatial join is sus, investigate

#### Notebooks

`bsl_processor_sigma.ipynb` - Obsolete notebook that combines individual bands to hsel

`bsl_reader_sigma.ipynb` - Obsolete notebook for spatially joining BSL and MW

`pickle_merge_actual.ipynb` - Not actually actual, final code in cpl\_stacker.py

`pickle_proto.ipynb` - Testing code to look at CPL data

### Discover scripts

These are the scripts from discover for dealing with raw data

`cpl_ingest.py` - Reads bsl netcdf and turns it into pickles

`data_ingest_all_v4.py` - Parses the raw ascii files and turns it into single day npy

`data_process.py` - process npy files, super hacky multiprocessing, tries to use as many cores as possible and when it runs out of memory tries again with less cores lol

`makeday.py` - This is what generates fulldata/ on gpu2

`gen_new_redo.py` - helper script that makes an empty pickle file, yeah..

`run.sh` - runs data\_ingest\_all\_v4.py

`run2.sh` - runs data\_process.py

`run3.sh` - runs makeday.py

Typical usage is running the run scripts in order run->run2->run3

#### Prototype Models

`atms_dg_q_cpl_v4.ipynb` - Prototype cpl model: atms+cpl q

`atms_dg_t_cpl_v4.ipynb` - Prototype cpl model: atms+cpl T

`hsel_dg_q_cpl_v4.ipynb` - Prototype cpl model: hsel+cpl q

`hsel_dg_t_cpl_v4.ipynb` - Prototype cpl model: hsel+cpl T

`atms_dg_q_all_v3.ipynb` - Prototype passive-only model: atms q

`atms_dg_t_all_v3.ipynb` - Prototype passive-only model: atms T

`hsel_dg_q_all_v3.ipynb` - Prototype passive-only model: hsel q

`hsel_dg_t_all_v2.ipynb` - Prototype passive-only model: hsel T


`hsel_ae.ipynb` - Prototype autoencoder notebook

`encoder_3_mae.keras.bz2` - Compressed autoencoder model used in proto hsel models


### CDF Gen

These are for manually generating cdfs given runids:

`cdf_maker.py` - Loads data and generates cdf, manually must set args

`makedf.sh` - runs cdf_maker to do a few in a batch

### Other

`notes.org` - Contains information on the raw data on discover and where to find it
