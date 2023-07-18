import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


#### IMPORTS #######
import os
import sys
import json
import numpy as np
from pathlib import Path
import pandas as pd
import time


# SpikeInterface
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.comparison as sc
import spikeinterface.qualitymetrics as sqm

# Tensorflow
import tensorflow as tf


base_path = Path("../../..")

##### DEFINE DATASETS AND FOLDERS #######
from sessions import all_sessions

n_jobs = 16

job_kwargs = dict(n_jobs=n_jobs, progress_bar=True, chunk_duration="1s")

data_folder = base_path / "data"
scratch_folder = base_path / "scratch"
results_folder = base_path / "results"


# DATASET_BUCKET = "s3://aind-benchmark-data/ephys-compression/aind-np2/"
DATASET_BUCKET = data_folder / "ephys-compression-benchmark"

DEBUG = False
NUM_DEBUG_SESSIONS = 2
DEBUG_DURATION = 20

##### DEFINE PARAMS #####
OVERWRITE = False
USE_GPU = True
FULL_INFERENCE = True

# Define training and testing constants (@Jad you can gradually increase this)


FILTER_OPTIONS = ["bp", "hp"]  # "hp", "bp", "no"

# DI params
pre_frame = 30
post_frame = 30
pre_post_omission = 1
desired_shape = (192, 2)

di_kwargs = dict(
    pre_frame=pre_frame,
    post_frame=post_frame,
    pre_post_omission=pre_post_omission,
    desired_shape=desired_shape,
)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "true":
            DEBUG = True
        else:
            DEBUG = False

    json_files = [p for p in data_folder.iterdir() if p.name.endswith(".json")]

    if len(json_files) > 0:
        print(f"Found {len(json_files)} JSON config")
        session_dict = {}
        # each json file contains a session to run
        for json_file in json_files:
            with open(json_file, "r") as f:
                d = json.load(f)
                probe = d["probe"]
                if probe not in session_dict:
                    session_dict[probe] = []
                session = d["session"]
                assert (
                    session in all_sessions[probe]
                ), f"{session} is not a valid session. Valid sessions for {probe} are:\n{all_sessions[probe]}"
                session_dict[probe].append(session)
    else:
        session_dict = all_sessions

    print(session_dict)

    if DEBUG:
        TRAINING_START_S = 0
        TRAINING_END_S = 0.2
        TESTING_START_S = 10
        TESTING_END_S = 10.05
        OVERWRITE = True
    else:
        TRAINING_START_S = 0
        TRAINING_END_S = 20
        TESTING_START_S = 70
        TESTING_END_S = 70.5
        OVERWRITE = False

    si.set_global_job_kwargs(**job_kwargs)

    print(f"Tensorflow GPU status: {tf.config.list_physical_devices('GPU')}")

    for probe, sessions in session_dict.items():
        print(f"Dataset {probe}")
        if DEBUG and len(sessions) > NUM_DEBUG_SESSIONS:
            sessions_to_run = sessions[:NUM_DEBUG_SESSIONS]
        else:
            sessions_to_run = sessions
        for session in sessions_to_run:
            print(f"\nAnalyzing session {session}\n")
            if str(DATASET_BUCKET).startswith("s3"):
                raw_data_folder = scratch_folder / "raw"
                raw_data_folder.mkdir(exist_ok=True)

                # download dataset
                dst_folder.mkdir(exist_ok=True)

                src_folder = f"{DATASET_BUCKET}{session}"

                cmd = f"aws s3 sync {src_folder} {dst_folder}"
                # aws command to download
                os.system(cmd)
            else:
                raw_data_folder = DATASET_BUCKET
                dst_folder = raw_data_folder / session

            if "np1" in dst_folder.name:
                probe = "NP1"
            else:
                probe = "NP2"

            recording_folder = dst_folder
            recording = si.load_extractor(recording_folder)
            if DEBUG:
                recording = recording.frame_slice(
                    start_frame=0,
                    end_frame=int(DEBUG_DURATION * recording.sampling_frequency),
                )

            results_dict = {}
            for filter_option in FILTER_OPTIONS:
                print(f"\tFilter option: {filter_option}")
                results_dict[filter_option] = {}
                # train DI models
                print(f"\t\tTraning DI")
                training_time = np.round(TRAINING_END_S - TRAINING_START_S, 3)
                testing_time = np.round(TESTING_END_S - TESTING_START_S, 3)
                model_name = f"{filter_option}_t{training_time}s_v{testing_time}s"

                # apply filter and zscore
                if filter_option == "hp":
                    recording_processed = spre.highpass_filter(recording)
                elif filter_option == "bp":
                    recording_processed = spre.bandpass_filter(recording)
                else:
                    recording_processed = recording
                recording_zscore = spre.zscore(recording_processed)

                # train model
                model_folder = results_folder / "models" / session / filter_option
                model_folder.parent.mkdir(parents=True, exist_ok=True)
                # Use SI function
                t_start_training = time.perf_counter()
                model_path = spre.train_deepinterpolation(
                    recording_zscore,
                    model_folder=model_folder,
                    model_name=model_name,
                    train_start_s=TRAINING_START_S,
                    train_end_s=TRAINING_END_S,
                    test_start_s=TESTING_START_S,
                    test_end_s=TESTING_END_S,
                    **di_kwargs,
                )
                t_stop_training = time.perf_counter()
                elapsed_time_training = np.round(t_stop_training - t_start_training, 2)
                print(f"\t\tElapsed time TRAINING: {elapsed_time_training}s")
