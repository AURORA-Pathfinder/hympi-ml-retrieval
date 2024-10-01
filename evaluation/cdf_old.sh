#!/bin/bash
set -e

# T
runid_atms=1fd48b59482c4f73ae543e8687332ecf
python cdf_maker.py $runid_atms 20060815 1 train atms
python cdf_maker.py $runid_atms 20060615 1 train atms
python cdf_maker.py $runid_atms 20060803 1 test atms

# q
runid_atms=1414c407e6924ea983b285cc87458eb7
python cdf_maker.py $runid_atms 20060815 2 train atms
python cdf_maker.py $runid_atms 20060615 2 train atms
python cdf_maker.py $runid_atms 20060803 2 test atms

# T
runid_hsel=63a5a1bc210546aa9645267b94983f55
python cdf_maker.py $runid_hsel 20060815 1 train hsel
python cdf_maker.py $runid_hsel 20060615 1 train hsel
python cdf_maker.py $runid_hsel 20060803 1 test hsel

# q
runid_hsel=fefc09bce1bc4ed3adb4cd602c711038
python cdf_maker.py $runid_hsel 20060815 2 train hsel
python cdf_maker.py $runid_hsel 20060615 2 train hsel
python cdf_maker.py $runid_hsel 20060803 2 test hsel

