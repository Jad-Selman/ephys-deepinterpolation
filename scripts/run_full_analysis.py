#### IMPORTS #######
import os
import numpy as np
from pathlib import Path
from numba import cuda 
import pandas as pd


# SpikeInterface
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.comparison as sc
import spikeinterface.qualitymetrics as sqm

from utils import train_di_model


##### DEFINE DATASETS AND FOLDERS #######

DATASET_BUCKET = "s3://aind-benchmark-data/ephys-compression/aind-np2/"

sessions = [
    "595262_2022-02-21_15-18-07_ProbeA",
    "602454_2022-03-22_16-30-03_ProbeB",
    "612962_2022-04-13_19-18-04_ProbeB",
    "612962_2022-04-14_17-17-10_ProbeC",
    "618197_2022-06-21_14-08-06_ProbeC",
    "618318_2022-04-13_14-59-07_ProbeB",
    "618384_2022-04-14_15-11-00_ProbeB",
    "621362_2022-07-14_11-19-36_ProbeA",
]
sessions = sessions[:1]
n_jobs = 16

data_folder = Path("../data")
results_folder = Path("../results")

DEBUG = True
DEBUG_DURATION = 20

##### DEFINE PARAMS #####
OVERWRITE = False
FULL_INFERENCE = True

# Define training and testing constants (@Jad you can gradually increase this)
if DEBUG:
    TRAINING_START_S = 0
    TRAINING_END_S = 0.5
    TESTING_START_S = 10
    TESTING_END_S = 10.05
else:
    TRAINING_START_S = 0
    TRAINING_END_S = 20
    TESTING_START_S = 70
    TESTING_END_S = 70.5

FILTER_OPTIONS = ["bp", "hp"] # "hp", "bp", "no"

# DI params
pre_frame = 30
post_frame = 30
pre_post_omission = 1
desired_shape = (192, 2)
inference_n_jobs = 1 # TODO: Jad - try more jobs
inference_chunk_duration = "50ms"

di_kwargs = dict(
    pre_frame=pre_frame,
    post_frame=post_frame,
    pre_post_omission=pre_post_omission,
    desired_shape=desired_shape,
    inference_n_jobs=inference_n_jobs,
    inference_chunk_duration=inference_chunk_duration
)

sorter_name = "kilosort2_5"
singularity_image = False

match_score = 0.7

#### START ####
raw_data_folder = data_folder / "raw"
raw_data_folder.mkdir(exist_ok=True)

session_level_results = pd.DataFrame(columns=['session', 'filter_option', 'di', "num_units", "sorting_path"])

unit_level_results = pd.DataFrame(columns=['session', 'filter_option', 'di', "unit_index",
                                           "unit_id_no_di", "unit_id_di"])

for session in sessions:
    print(f"Analyzing session {session}")

    # download dataset
    dst_folder = (raw_data_folder / session)
    dst_folder.mkdir(exist_ok=True)

    src_folder = f"{DATASET_BUCKET}{session}"
    
    cmd = f"aws s3 sync {src_folder} {dst_folder}"
    # aws command to download
    os.system(cmd)

    recording_folder = dst_folder
    recording = si.load_extractor(recording_folder)
    if DEBUG:
        recording = recording.frame_slice(start_frame=0, end_frame=int(DEBUG_DURATION * recording.sampling_frequency))

    results_dict = {}
    for filter_option in FILTER_OPTIONS:
        print(f"\tFilter option: {filter_option}")
        results_dict[filter_option] = {}
        # train DI models
        print(f"\t\tTraning DI")
        training_time = np.round(TRAINING_END_S - TRAINING_START_S, 3)
        testing_time = np.round(TESTING_END_S - TESTING_START_S, 3)
        model_name = f"{filter_option}_t{training_time}s_v{testing_time}s"
        recording_no_di, recording_di = train_di_model(recording, session, filter_option, 
                                                       TRAINING_START_S, TRAINING_END_S, 
                                                       TESTING_START_S, TESTING_END_S, 
                                                       data_folder, FULL_INFERENCE, model_name, 
                                                       di_kwargs, overwrite=OVERWRITE)
        results_dict[filter_option]["recording_no_di"] = recording_no_di
        results_dict[filter_option]["recording_di"] = recording_di

        # release GPU memory
        device = cuda.get_current_device()
        device.reset()

        # run spike sorting
        sorting_output_folder = data_folder / "sortings" / session
        sorting_output_folder.mkdir(parents=True, exist_ok=True)
        
        recording_no_di = results_dict[filter_option]["recording_no_di"]
        if (sorting_output_folder / f"no_di_{model_name}").is_dir() and not OVERWRITE:
            print("\t\tLoading NO DI sorting")
            sorting_no_di = si.load_extractor(sorting_output_folder / f"no_di_{model_name}")
        else:
            print(f"\t\tSpike sorting NO DI with {sorter_name}")
            sorting_no_di = ss.run_sorter(sorter_name, recording=recording_no_di, 
                                        n_jobs=n_jobs, verbose=True, singularity_image=singularity_image)
            sorting_no_di = sorting_no_di.save(folder=sorting_output_folder / f"no_di_{model_name}")
        results_dict[filter_option]["sorting_no_di"] = sorting_no_di

        recording_di = results_dict[filter_option]["recording_di"]
        if (sorting_output_folder / f"di_{model_name}").is_dir() and not OVERWRITE:
            print("\t\tLoading DI sorting")
            sorting_di = si.load_extractor(sorting_output_folder / f"di_{model_name}")
        else:
            print(f"\t\tSpike sorting DI with {sorter_name}")
            sorting_di = ss.run_sorter(sorter_name, recording=recording_di, 
                                    n_jobs=n_jobs, verbose=True, singularity_image=singularity_image)
            sorting_di = sorting_di.save(folder=sorting_output_folder / f"di_{model_name}")
        results_dict[filter_option]["sorting_di"] = sorting_di

        # TODO: Jad - compute waveforms and quality metrics (https://spikeinterface.readthedocs.io/en/latest/how_to/get_started.html)

        ## add entries to session-level results
        session_level_results.append({"session": session, "filter_option": filter_option,
                                      "di": False, "num_units": len(sorting_no_di.unit_ids),
                                      "sorting_path": str((sorting_output_folder / f"no_di_{model_name}").absolute())},
                                     ignore_index=True)
        session_level_results.append({"session": session, "filter_option": filter_option,
                                      "di": True, "num_units": len(sorting_di.unit_ids),
                                      "sorting_path": str((sorting_output_folder / f"di_{model_name}").absolute())},
                                     ignore_index=True)

        # compare outputs
        print("\t\tComparing sortings")
        cmp = sc.compare_two_sorters(sorting1=sorting_no_di, sorting2=sorting_di,
                                     sorting1_name="no_di", sorting2_name="di",
                                     match_score=match_score)
        matched_units = cmp.get_matching()[0]
        matched_unit_ids_no_di = matched_units.index.values.astype(int)
        matched_unit_ids_di = matched_units.values.astype(int)
        matched_units_valid = matched_unit_ids_di != -1
        matched_unit_ids_no_di = matched_unit_ids_no_di[matched_units_valid]
        matched_unit_ids_di = matched_unit_ids_di[matched_units_valid]
        sorting_no_di_matched = sorting_no_di.select_units(unit_ids=matched_unit_ids_no_di)
        sorting_di_matched = sorting_di.select_units(unit_ids=matched_unit_ids_di)

        waveforms_folder = data_folder / "waveforms" / session
        waveforms_folder.mkdir(exist_ok=True, parents=True)

        if (waveforms_folder / f"no_di_{model_name}").is_dir() and not OVERWRITE:
            print("\t\tLoad NO DI waveforms")
            we_no_di = si.load_waveforms(waveforms_folder / f"no_di_{model_name}")
        else:
            print("\t\tCompute NO DI waveforms")
            we_no_di = si.extract_waveforms(recording_no_di, sorting_no_di_matched, 
                                            folder=waveforms_folder / f"no_di_{model_name}",
                                            n_jobs=n_jobs, overwrite=True)
        results_dict[filter_option]["we_no_di"] = we_no_di
        
        if (waveforms_folder / f"di_{model_name}").is_dir() and not OVERWRITE:
            print("\t\tLoad DI waveforms")
            we_di = si.load_waveforms(waveforms_folder / f"di_{model_name}")
        else:
            print("\t\tCompute DI waveforms")
            we_di = si.extract_waveforms(recording_di, sorting_di_matched, 
                                         folder=waveforms_folder / f"di_{model_name}",
                                         n_jobs=n_jobs, overwrite=True)
        results_dict[filter_option]["we_di"] = we_di

        # compute metrics
        if we_no_di.is_extension("quality_metrics") and not OVERWRITE:
            print("\t\tLoad NO DI metrics")
            qm_no_di = we_no_di.load_extension("quality_metrics").get_data()
        else:
            print("\t\tCompute NO DI metrics")
            qm_no_di = sqm.compute_quality_metrics(we_no_di)
        results_dict[filter_option]["qm_no_di"] = qm_no_di

        if we_di.is_extension("quality_metrics") and not OVERWRITE:
            print("\t\tLoad DI metrics")
            qm_di = we_di.load_extension("quality_metrics").get_data()
        else:
            print("\t\tCompute DI metrics")
            qm_di = sqm.compute_quality_metrics(we_di)
        results_dict[filter_option]["qm_di"] = qm_di

        ## add entries to unit-level results

results_folder.mkdir(exist_ok=True)
session_level_results.to_csv(results_folder / "session-results.csv")
unit_level_results.to_csv(results_folder / "unit-results.csv")
