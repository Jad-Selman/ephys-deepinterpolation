import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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


base_path = Path("..").resolve()

##### DEFINE DATASETS AND FOLDERS #######
from sessions import all_sessions_exp, all_sessions_sim

n_jobs = -1

job_kwargs = dict(n_jobs=n_jobs, progress_bar=True, chunk_duration="1s")

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
inference_n_jobs = -1
inference_chunk_duration = "1s"
inference_predict_workers = 1
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
            OVERWRITE = True
        else:
            DEBUG = False
            OVERWRITE = False

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

    si.set_global_job_kwargs(**job_kwargs)

    print(f"Tensorflow GPU status: {tf.config.list_physical_devices('GPU')}")

    #### START ####
    probe_models_folders = [p for p in data_folder.iterdir() if "model_" in p.name and p.is_dir()]

    if len(probe_models_folders) > 0:
        data_model_folder = data_folder
    else:
        data_model_subfolders = []
        for p in data_folder.iterdir():
            if p.is_dir() and len([pp for pp in p.iterdir() if "model_" in pp.name and pp.is_dir()]) > 0:
                data_model_subfolders.append(p)
        data_model_folder = data_model_subfolders[0]

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

            if DEBUG:
                recording = recording.frame_slice(
                    start_frame=0,
                    end_frame=int(DEBUG_DURATION * recording.sampling_frequency),
                )
            print(f"\t{recording}")

            for filter_option in FILTER_OPTIONS:
                print(f"\tFilter option: {filter_option}")

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
                recording_zscore_bin = recording_zscore.save(folder=scratch_folder / "recording_zscored")

                # train model
                model_folder = data_model_folder / f"model_{dataset_name}_{session_name}_{filter_option}"
                model_path = [p for p in model_folder.iterdir() if p.name.endswith("model.h5")][0]
                # full inference
                output_folder = results_folder / f"deepinterpolated_{dataset_name}_{session_name}_{filter_option}"
                if OVERWRITE and output_folder.is_dir():
                    shutil.rmtree(output_folder)

                if not output_folder.is_dir():
                    t_start_inference = time.perf_counter()
                    output_folder.parent.mkdir(exist_ok=True, parents=True)
                    recording_di = spre.deepinterpolate(
                        recording_zscore_bin,
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
                processed_folder = results_folder / f"processed_{dataset_name}_{session_name}_{filter_option}"
                processed_folder.mkdir(exist_ok=True, parents=True)
                recording_processed.dump_to_json(processed_folder / "processed.json", relative_to=results_folder)
                recording_di.dump_to_json(processed_folder / f"deepinterpolated.json", relative_to=results_folder)

    for json_file in json_files:
        shutil.copy(json_file, results_folder)
