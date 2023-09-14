import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


#### IMPORTS #######
import os
import sys
import numpy as np
from pathlib import Path
from numba import cuda
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
n_jobs = 16

job_kwargs = dict(n_jobs=n_jobs, progress_bar=True, chunk_duration="1s")

data_folder = base_path / "data"
scratch_folder = base_path / "scratch"
results_folder = base_path / "results"


# DATASET_BUCKET = "s3://aind-benchmark-data/ephys-compression/aind-np2/"
DATASET_BUCKET = data_folder / "ephys-compression-benchmark" / "aind-np2"

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
inference_n_jobs = 4
inference_chunk_duration = "100ms"
inference_memory_gpu = 2000 #MB

di_kwargs = dict(
    pre_frame=pre_frame,
    post_frame=post_frame,
    pre_post_omission=pre_post_omission,
    desired_shape=desired_shape,
)

sorter_name = "pykilosort"
singularity_image = False

match_score = 0.7


if __name__ == "__main__":
    if len(sys.argv) == 3:
        if sys.argv[1] == "true":
            DEBUG = True
        else:
            DEBUG = False
        if sys.argv[2] != "all":
            sessions = [sys.argv[2]]

    if DEBUG:
        TRAINING_START_S = 0
        TRAINING_END_S = 0.2
        TESTING_START_S = 10
        TESTING_END_S = 10.05
        if len(sessions) > NUM_DEBUG_SESSIONS:
            sessions = sessions[:NUM_DEBUG_SESSIONS]
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
    session_level_results = pd.DataFrame(
        columns=[
            "session",
            "probe",
            "filter_option",
            "num_units",
            "num_units_di",
            "sorting_path",
            "sorting_path_di",
            "num_match",
        ]
    )

    unit_level_results_columns = [
        "session",
        "probe",
        "filter_option",
        "unit_id",
        "unit_id_di",
    ]
    unit_level_results = None

    sessions = sessions[:2]

    for session in sessions:
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
            # full inference
            output_folder = (
                results_folder / "deepinterpolated" / session / filter_option
            )
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
                    use_gpu=USE_GPU
                )
                recording_di = recording_di.save(
                    folder=output_folder,
                    n_jobs=inference_n_jobs,
                    chunk_duration=inference_chunk_duration,
                )
                t_stop_inference = time.perf_counter()
                elapsed_time_inference = np.round(
                    t_stop_inference - t_start_inference, 2
                )
                print(f"\t\tElapsed time INFERENCE: {elapsed_time_inference}s")
            else:
                print("\t\tLoading existing folder")
                recording_di = si.load_extractor(output_folder)
            # apply inverse z-scoring
            inverse_gains = 1 / recording_zscore.gain
            inverse_offset = -recording_zscore.offset * inverse_gains
            recording_di_inverse_zscore = spre.scale(
                recording_di, gain=inverse_gains, offset=inverse_offset, dtype="float"
            )

            results_dict[filter_option]["recording_no_di"] = recording_processed
            results_dict[filter_option]["recording_di"] = recording_di_inverse_zscore

            # run spike sorting
            sorting_output_folder = (
                results_folder / "sortings" / session / filter_option
            )
            sorting_output_folder.mkdir(parents=True, exist_ok=True)

            recording_no_di = results_dict[filter_option]["recording_no_di"]
            if (
                sorting_output_folder / f"no_di_{model_name}"
            ).is_dir() and not OVERWRITE:
                print("\t\tLoading NO DI sorting")
                sorting_no_di = si.load_extractor(sorting_output_folder / "sorting")
            else:
                print(f"\t\tSpike sorting NO DI with {sorter_name}")
                sorting_no_di = ss.run_sorter(
                    sorter_name,
                    recording=recording_no_di,
                    n_jobs=n_jobs,
                    verbose=True,
                    singularity_image=singularity_image,
                )
                sorting_no_di = sorting_no_di.save(
                    folder=sorting_output_folder / "sorting"
                )
            results_dict[filter_option]["sorting_no_di"] = sorting_no_di

            recording_di = results_dict[filter_option]["recording_di"]
            if (sorting_output_folder / f"di_{model_name}").is_dir() and not OVERWRITE:
                print("\t\tLoading DI sorting")
                sorting_di = si.load_extractor(sorting_output_folder / "sorting_di")
            else:
                print(f"\t\tSpike sorting DI with {sorter_name}")
                sorting_di = ss.run_sorter(
                    sorter_name,
                    recording=recording_di,
                    n_jobs=n_jobs,
                    verbose=True,
                    singularity_image=singularity_image,
                )
                sorting_di = sorting_di.save(
                    folder=sorting_output_folder / "sorting_di"
                )
            results_dict[filter_option]["sorting_di"] = sorting_di

            # compare outputs
            print("\t\tComparing sortings")
            comp = sc.compare_two_sorters(
                sorting1=sorting_no_di,
                sorting2=sorting_di,
                sorting1_name="no_di",
                sorting2_name="di",
                match_score=match_score,
            )
            matched_units = comp.get_matching()[0]
            matched_unit_ids_no_di = matched_units.index.values.astype(int)
            matched_unit_ids_di = matched_units.values.astype(int)
            matched_units_valid = matched_unit_ids_di != -1
            matched_unit_ids_no_di = matched_unit_ids_no_di[matched_units_valid]
            matched_unit_ids_di = matched_unit_ids_di[matched_units_valid]
            sorting_no_di_matched = sorting_no_di.select_units(
                unit_ids=matched_unit_ids_no_di
            )
            sorting_di_matched = sorting_di.select_units(unit_ids=matched_unit_ids_di)

            ## add entries to session-level results
            new_row = {
                "session": session,
                "filter_option": filter_option,
                "probe": probe,
                "num_units": len(sorting_no_di.unit_ids),
                "num_units_di": len(sorting_di.unit_ids),
                "num_match": len(sorting_no_di_matched.unit_ids),
                "sorting_path": str(
                    (sorting_output_folder / "sorting").relative_to(results_folder)
                ),
                "sorting_path_di": str(
                    (sorting_output_folder / "sorting_di_{model_name}").relative_to(
                        results_folder
                    )
                ),
            }
            session_level_results = pd.concat(
                [session_level_results, pd.DataFrame([new_row])], ignore_index=True
            )

            print(
                f"\n\t\tNum units: {new_row['num_units']} - Num units DI: {new_row['num_units_di']} - Num match: {new_row['num_match']}"
            )

            # waveforms
            waveforms_folder = results_folder / "waveforms" / session / filter_option
            waveforms_folder.mkdir(exist_ok=True, parents=True)

            if (waveforms_folder / f"no_di_{model_name}").is_dir() and not OVERWRITE:
                print("\t\tLoad NO DI waveforms")
                we_no_di = si.load_waveforms(waveforms_folder / f"no_di_{model_name}")
            else:
                print("\t\tCompute NO DI waveforms")
                we_no_di = si.extract_waveforms(
                    recording_no_di,
                    sorting_no_di_matched,
                    folder=waveforms_folder / f"no_di_{model_name}",
                    n_jobs=n_jobs,
                    overwrite=True,
                )
            results_dict[filter_option]["we_no_di"] = we_no_di

            if (waveforms_folder / f"di_{model_name}").is_dir() and not OVERWRITE:
                print("\t\tLoad DI waveforms")
                we_di = si.load_waveforms(waveforms_folder / f"di_{model_name}")
            else:
                print("\t\tCompute DI waveforms")
                we_di = si.extract_waveforms(
                    recording_di,
                    sorting_di_matched,
                    folder=waveforms_folder / f"di_{model_name}",
                    n_jobs=n_jobs,
                    overwrite=True,
                )
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
            if unit_level_results is None:
                for metric in qm_no_di.columns:
                    unit_level_results_columns.append(metric)
                    unit_level_results_columns.append(f"{metric}_di")
                unit_level_results = pd.DataFrame(columns=unit_level_results_columns)

            new_rows = {
                "session": [session] * len(qm_no_di),
                "probe": [probe] * len(qm_no_di),
                "filter_option": [filter_option] * len(qm_no_di),
                "unit_id": we_no_di.unit_ids,
                "unit_id_di": we_di.unit_ids,
            }
            for metric in qm_no_di.columns:
                new_rows[metric] = qm_no_di[metric].values
                new_rows[f"{metric}_di"] = qm_di[metric].values
            # append new entries
            unit_level_results = pd.concat(
                [unit_level_results, pd.DataFrame(new_rows)], ignore_index=True
            )

    results_folder.mkdir(exist_ok=True)
    session_level_results.to_csv(results_folder / "session-results.csv")
    unit_level_results.to_csv(results_folder / "unit-results.csv")
