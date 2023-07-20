import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


#### IMPORTS #######
import os
import sys
import json
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


base_path = Path("..").resolve()

##### DEFINE DATASETS AND FOLDERS #######
from sessions import all_sessions

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
OVERWRITE = False


# Define training and testing constants
FILTER_OPTIONS = ["bp", "hp"]  # "hp", "bp", "no"


sorter_name = "pykilosort"
singularity_image = False
match_score = 0.7


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "true":
            DEBUG = True
            OVERWRITE = True
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

    si.set_global_job_kwargs(**job_kwargs)

    #### START ####
    probe_processed_folders = [p for p in data_folder.iterdir() if "processed_" in p.name and p.is_dir()]

    if len(probe_processed_folders) > 0:
        processed_folder = data_folder
    else:
        data_subfolders = [p for p in data_folder.iterdir() if p.is_dir()]
        processed_folder = data_subfolders[0]

    for probe, sessions in session_dict.items():

        print(f"Dataset {probe}")
        for session in sessions:
            print(f"\nAnalyzing session {session}\n")
            dataset_name, session_name = session.split("/")

            session_level_results = pd.DataFrame(
                columns=[
                    "dataset",
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
                "dataset",
                "session",
                "probe",
                "filter_option",
                "unit_id",
                "unit_id_di",
                "agreement_score",
            ]
            unit_level_results = None

            for filter_option in FILTER_OPTIONS:
                print(f"\tFilter option: {filter_option}")

                # load recordings
                # save processed json
                processed_json_folder = processed_folder / f"processed_{dataset_name}_{session_name}_{filter_option}"
                recording = si.load_extractor(processed_json_folder / "processed.json", base_folder=data_folder)
                recording_di = si.load_extractor(
                    processed_json_folder / "deepinterpolated.json", base_folder=processed_folder
                )

                # run spike sorting
                sorting_output_folder = results_folder / f"sorting_{dataset_name}_{session_name}_{filter_option}"
                sorting_output_folder.mkdir(parents=True, exist_ok=True)

                if (sorting_output_folder / "sorting").is_dir() and not OVERWRITE:
                    print("\t\tLoading NO DI sorting")
                    sorting = si.load_extractor(sorting_output_folder / "sorting")
                else:
                    print(f"\t\tSpike sorting NO DI with {sorter_name}")
                    sorting = ss.run_sorter(
                        sorter_name,
                        recording=recording,
                        output_folder=scratch_folder / session / filter_option / "no_di",
                        n_jobs=n_jobs,
                        verbose=True,
                        singularity_image=singularity_image,
                    )
                    sorting = sorting.save(folder=sorting_output_folder / "sorting")

                if (sorting_output_folder / "sorting_di").is_dir() and not OVERWRITE:
                    print("\t\tLoading DI sorting")
                    sorting_di = si.load_extractor(sorting_output_folder / "sorting_di")
                else:
                    print(f"\t\tSpike sorting DI with {sorter_name}")
                    sorting_di = ss.run_sorter(
                        sorter_name,
                        recording=recording_di,
                        output_folder=scratch_folder / session / filter_option / "di",
                        n_jobs=n_jobs,
                        verbose=True,
                        singularity_image=singularity_image,
                    )
                    sorting_di = sorting_di.save(folder=sorting_output_folder / "sorting_di")

                # compare outputs
                print("\t\tComparing sortings")
                comp = sc.compare_two_sorters(
                    sorting1=sorting,
                    sorting2=sorting_di,
                    sorting1_name="no_di",
                    sorting2_name="di",
                    match_score=match_score,
                )
                matched_units = comp.get_matching()[0]
                matched_unit_ids = matched_units.index.values.astype(int)
                matched_unit_ids_di = matched_units.values.astype(int)
                matched_units_valid = matched_unit_ids_di != -1
                matched_unit_ids = matched_unit_ids[matched_units_valid]
                matched_unit_ids_di = matched_unit_ids_di[matched_units_valid]
                sorting_matched = sorting.select_units(unit_ids=matched_unit_ids)
                sorting_di_matched = sorting_di.select_units(unit_ids=matched_unit_ids_di)

                ## add entries to session-level results
                new_row = {
                    "dataset": dataset_name,
                    "session": session_name,
                    "filter_option": filter_option,
                    "probe": probe,
                    "num_units": len(sorting.unit_ids),
                    "num_units_di": len(sorting_di.unit_ids),
                    "num_match": len(sorting_matched.unit_ids),
                    "sorting_path": str((sorting_output_folder / "sorting").relative_to(results_folder)),
                    "sorting_path_di": str((sorting_output_folder / "sorting_di_").relative_to(results_folder)),
                }
                session_level_results = pd.concat([session_level_results, pd.DataFrame([new_row])], ignore_index=True)

                print(
                    f"\n\t\tNum units: {new_row['num_units']} - Num units DI: {new_row['num_units_di']} - Num match: {new_row['num_match']}"
                )

                # waveforms
                waveforms_folder = results_folder / f"waveforms_{dataset_name}_{session_name}_{filter_option}"
                waveforms_folder.mkdir(exist_ok=True, parents=True)

                if (waveforms_folder / "waveforms").is_dir() and not OVERWRITE:
                    print("\t\tLoad NO DI waveforms")
                    we = si.load_waveforms(waveforms_folder / "waveforms")
                else:
                    print("\t\tCompute NO DI waveforms")
                    we = si.extract_waveforms(
                        recording,
                        sorting_matched,
                        folder=waveforms_folder / "waveforms",
                        n_jobs=n_jobs,
                        overwrite=True,
                    )

                if (waveforms_folder / "waveforms_di").is_dir() and not OVERWRITE:
                    print("\t\tLoad DI waveforms")
                    we_di = si.load_waveforms(waveforms_folder / "waveforms_di")
                else:
                    print("\t\tCompute DI waveforms")
                    we_di = si.extract_waveforms(
                        recording_di,
                        sorting_di_matched,
                        folder=waveforms_folder / "waveforms_di",
                        n_jobs=n_jobs,
                        overwrite=True,
                    )

                # compute metrics
                if we.is_extension("quality_metrics") and not OVERWRITE:
                    print("\t\tLoad NO DI metrics")
                    qm = we.load_extension("quality_metrics").get_data()
                else:
                    print("\t\tCompute NO DI metrics")
                    qm = sqm.compute_quality_metrics(we)

                if we_di.is_extension("quality_metrics") and not OVERWRITE:
                    print("\t\tLoad DI metrics")
                    qm_di = we_di.load_extension("quality_metrics").get_data()
                else:
                    print("\t\tCompute DI metrics")
                    qm_di = sqm.compute_quality_metrics(we_di)

                ## add entries to unit-level results
                if unit_level_results is None:
                    for metric in qm.columns:
                        unit_level_results_columns.append(metric)
                        unit_level_results_columns.append(f"{metric}_di")
                    unit_level_results = pd.DataFrame(columns=unit_level_results_columns)

                new_rows = {
                    "dataset": [dataset_name] * len(qm),
                    "session": [session_name] * len(qm),
                    "probe": [probe] * len(qm),
                    "filter_option": [filter_option] * len(qm),
                    "unit_id": we.unit_ids,
                    "unit_id_di": we_di.unit_ids,
                }
                agreement_scores = []
                for i in range(len(we.unit_ids)):
                    agreement_scores.append(comp.agreement_scores.at[we.unit_ids[i], we_di.unit_ids[i]])
                new_rows["agreement_score"] = agreement_scores
                for metric in qm.columns:
                    new_rows[metric] = qm[metric].values
                    new_rows[f"{metric}_di"] = qm_di[metric].values
                # append new entries
                unit_level_results = pd.concat([unit_level_results, pd.DataFrame(new_rows)], ignore_index=True)

            session_level_results.to_csv(results_folder / f"{dataset_name}-{session_name}-sessions.csv", index=False)
            unit_level_results.to_csv(results_folder / f"{dataset_name}-{session_name}-units.csv", index=False)
