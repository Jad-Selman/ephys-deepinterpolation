import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


#### IMPORTS #######
import sys
import shutil
import json
import numpy as np
from pathlib import Path
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

# runs from "codes"
base_path = Path("..")

##### DEFINE DATASETS AND FOLDERS #######
from sessions import all_sessions_exp, all_sessions_sim

data_folder = base_path / "data"
scratch_folder = base_path / "scratch"
results_folder = base_path / "results"


if (data_folder / "ephys-compression-benchmark").is_dir():
    DATASET_FOLDER = data_folder / "ephys-compression-benchmark"
    all_sessions = all_sessions_exp
    data_type = "exp"
elif (data_folder / "MEArec-NP-recordings").is_dir():
    DATASET_FOLDER = data_folder / "MEArec-NP-recordings"
    all_sessions = all_sessions_sim
    data_type = "sim"
else:
    raise Exception("Could not find dataset folder")


n_jobs = 16
job_kwargs = dict(n_jobs=n_jobs, progress_bar=True, chunk_duration="1s")

DEBUG = False
NUM_DEBUG_SESSIONS = 2
DEBUG_DURATION = 20

##### DEFINE PARAMS #####
OVERWRITE = False
USE_GPU = True
STEPS_PER_EPOCH = 100

# Define training and testing constants
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
            STEPS_PER_EPOCH = 10
        else:
            DEBUG = False

    json_files = [p for p in data_folder.iterdir() if p.name.endswith(".json")]
    print(f"Found {len(json_files)} JSON config: {json_files}")
    if len(json_files) > 0:
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
        TRAINING_START_S = 10
        TRAINING_END_S = None
        TRAINING_DURATION_S = 1
        TESTING_START_S = 0
        TESTING_END_S = 10
        TESTING_DURATION_S = 0.05
        OVERWRITE = True
    else:
        TRAINING_START_S = 10
        TRAINING_END_S = None
        TRAINING_DURATION_S = 600
        TESTING_START_S = 0
        TESTING_END_S = 10
        TESTING_DURATION_S = 0.1
        OVERWRITE = False

    si.set_global_job_kwargs(**job_kwargs)

    available_gpus = tf.config.list_physical_devices("GPU")
    print(f"Tensorflow GPU status: {available_gpus}")
    nb_gpus = len(available_gpus)
    if nb_gpus > 1:
        print("Use 1 GPU only!")
        nb_gpus = 1

    for probe, sessions in session_dict.items():
        print(f"Dataset {probe}")

        for session in sessions:
            print(f"\nAnalyzing session {session}\n")
            dataset_name, session_name = session.split("/")

            if data_type == "exp":
                recording = si.load_extractor(DATASET_FOLDER / session)
            else:
                recording, _ = se.read_mearec(DATASET_FOLDER / session)
                session_name = session_name.split(".")[0]
                recording = spre.depth_order(recording)

            if DEBUG:
                recording = recording.frame_slice(
                    start_frame=0,
                    end_frame=int(DEBUG_DURATION * recording.sampling_frequency),
                )
            print(f"\t{recording}")
            if TRAINING_END_S is None:
                TRAINING_END_S = recording.get_total_duration()

            for filter_option in FILTER_OPTIONS:
                print(f"\tFilter option: {filter_option}")
                recording_name = f"{dataset_name}_{session_name}_{filter_option}"
                # train DI models
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

                if data_type == "sim":
                    recording_processed = spre.depth_order(recording_processed)

                recording_zscore = spre.zscore(recording_processed)
                # This speeds things up a lot
                recording_zscore_bin = recording_zscore.save(
                    folder=scratch_folder / f"recording_zscored_{recording_name}"
                )

                # train model
                model_folder = results_folder / f"model_{recording_name}"
                model_folder.parent.mkdir(parents=True, exist_ok=True)
                # Use SI function
                t_start_training = time.perf_counter()
                model_path = spre.train_deepinterpolation(
                    recording_zscore_bin,
                    model_folder=model_folder,
                    model_name=model_name,
                    train_start_s=TRAINING_START_S,
                    train_end_s=TRAINING_END_S,
                    train_duration_s=TRAINING_DURATION_S,
                    test_start_s=TESTING_START_S,
                    test_end_s=TESTING_END_S,
                    test_duration_s=TESTING_DURATION_S,
                    verbose=False,
                    nb_gpus=nb_gpus,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    **di_kwargs,
                )
                t_stop_training = time.perf_counter()
                elapsed_time_training = np.round(t_stop_training - t_start_training, 2)
                print(f"\t\tElapsed time TRAINING {session}-{filter_option}: {elapsed_time_training}s")

    for json_file in json_files:
        print(f"Copying JSON file: {json_file.name} to {results_folder}")
        shutil.copy(json_file, results_folder)
