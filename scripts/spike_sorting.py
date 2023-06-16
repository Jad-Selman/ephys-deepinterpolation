import sys
sys.path.append("../src")

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
from pathlib import Path

# example of data generation in spike interface
folder_path = "/home/buccino/codes/ephys-deepinterpolation/data/neuropixels2.0_rescaled_recording/recording_rescaled_hp_di"
#recording = se.read_openephys(folder_path)


recording = si.load_extractor(folder_path)

print(recording)

sorter_name = "kilosort2_5"

sorting_ks25 = ss.run_sorter(sorter_name, recording=recording, n_jobs=16, verbose=True, singularity_image=True)
sorting_ks25= sorting_ks25.save(folder= Path("../data/sorting_outputs/kilosort25/")/f"{sorter_name}_di_bp_t20s_v0.5s" )

print(sorting_ks25)
