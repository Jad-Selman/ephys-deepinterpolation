import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


#### IMPORTS #######
import os
import sys
import shutil
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


os.environ["OPENBLAS_NUM_THREADS"] = "1"


base_path = Path("../../..").resolve()

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
# play around with these
inference_n_jobs = 16
inference_chunk_duration = "500ms"
inference_predict_workers = 8
inference_memory_gpu = 2000  # MB

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

    #### START ####

    if (data_folder / "models").is_dir():
        data_model_folder = data_folder / "models"
    else:
        data_subfolders = [p for p in data_folder.iterdir() if (p / "models").is_dir()]
        assert len(data_subfolders) == 1
        data_model_folder = data_subfolders[0] / "models"

    for probe, sessions in session_dict.items():
        print(f"Dataset {probe}")
        for session in sessions:
            print(f"\nAnalyzing session {session}\n")
            dataset_name, session_name = session.split("/")

            if str(DATASET_BUCKET).startswith("s3"):
                raw_data_folder = scratch_folder / "raw"
                raw_data_folder.mkdir(exist_ok=True)
                dst_folder = raw_data_folder / session

                # download dataset
                dst_folder.mkdir(exist_ok=True)

                src_folder = f"{DATASET_BUCKET}{session}"

                cmd = f"aws s3 sync --no-sign-request {src_folder} {dst_folder}"
                # aws command to download
                os.system(cmd)
            else:
                raw_data_folder = DATASET_BUCKET
                dst_folder = raw_data_folder / session

            recording_folder = dst_folder
            recording = si.load_extractor(recording_folder)
            if DEBUG:
                recording = recording.frame_slice(
                    start_frame=0,
                    end_frame=int(DEBUG_DURATION * recording.sampling_frequency),
                )

            for filter_option in FILTER_OPTIONS:
                print(f"\tFilter option: {filter_option}")
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
                model_folder = data_model_folder / session / filter_option
                model_path = [
                    p for p in model_folder.iterdir() if p.name.endswith("model.h5") and filter_option in p.name
                ][0]
                # full inference
                output_folder = results_folder / "deepinterpolated" / session / filter_option
                if OVERWRITE and output_folder.is_dir():
                    shutil.rmtree(output_folder)

                if not output_folder.is_dir():
                    t_start_inference = time.perf_counter()
                    output_folder.parent.mkdir(exist_ok=True, parents=True)
                    recording_di = spre.deepinterpolate(
                        recording_zscore,
                        model_path=model_path,
                        pre_frame=pre_frame,
                        post_frame=post_frame,
                        pre_post_omission=pre_post_omission,
                        memory_gpu=inference_memory_gpu,
                        predict_workers=inference_predict_workers,
                        use_gpu=USE_GPU,
                    )
                    recording_di = recording_di.save(
                        folder=output_folder,
                        n_jobs=inference_n_jobs,
                        chunk_duration=inference_chunk_duration,
                    )
                    t_stop_inference = time.perf_counter()
                    elapsed_time_inference = np.round(t_stop_inference - t_start_inference, 2)
                    print(f"\t\tElapsed time INFERENCE: {elapsed_time_inference}s")
                else:
                    print("\t\tLoading existing folder")
                    recording_di = si.load_extractor(output_folder)
                # apply inverse z-scoring
                inverse_gains = 1 / recording_zscore.gain
                inverse_offset = -recording_zscore.offset * inverse_gains
                recording_di = spre.scale(recording_di, gain=inverse_gains, offset=inverse_offset, dtype="float")

                # save processed json
                processed_folder = results_folder / "processed" / session / filter_option
                processed_folder.mkdir(exist_ok=True, parents=True)
                recording_processed.dump_to_json(processed_folder / "processed.json", relative_to=results_folder)
                recording_di.dump_to_json(processed_folder / f"deepinterpolated.json", relative_to=results_folder)

    for json_file in json_files:
        shutil.copy(json_file, results_folder)
