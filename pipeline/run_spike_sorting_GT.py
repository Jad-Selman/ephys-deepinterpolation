import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


#### IMPORTS #######
import sys
import json
from pathlib import Path
import pandas as pd


# SpikeInterface
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.curation as scur
import spikeinterface.comparison as sc
import spikeinterface.qualitymetrics as sqm


base_path = Path("..").resolve()

##### DEFINE DATASETS AND FOLDERS #######
from sessions import all_sessions_sim as all_sessions

n_jobs = 16

job_kwargs = dict(n_jobs=n_jobs, progress_bar=True, chunk_duration="1s")

data_folder = base_path / "data"
scratch_folder = base_path / "scratch"
results_folder = base_path / "results"

DATASET_FOLDER = data_folder / "MEArec-NP-recordings"

OVERWRITE = False

# Define training and testing constants
FILTER_OPTIONS = ["bp", "hp"]  # "hp", "bp", "no"


sorter_name = "pykilosort"
singularity_image = False
match_score = 0.7


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "true":
            OVERWRITE = True
        else:
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

            recording_gt, sorting_gt = se.read_mearec(DATASET_FOLDER / session)
            session_name = session_name.split(".")[0]

            session_level_results = None
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

                # DEBUG mode
                if recording.get_num_samples() < recording_gt.get_num_samples():
                    print("DEBUG MODE: slicing GT")
                    sorting_gt = sorting_gt.frame_slice(start_frame=0, end_frame=recording.get_num_samples())

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
                    sorting = scur.remove_excess_spikes(sorting, recording)
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
                    sorting_di = scur.remove_excess_spikes(sorting_di, recording_di)
                    sorting_di = sorting_di.save(folder=sorting_output_folder / "sorting_di")

                # compare to GT
                print("\tRunning comparison")
                cmp = sc.compare_sorter_to_ground_truth(sorting_gt, sorting, exhaustive_gt=True)
                cmp_di = sc.compare_sorter_to_ground_truth(sorting_gt, sorting_di, exhaustive_gt=True)

                perf_avg = cmp.get_performance(method="pooled_with_average")
                perf_avg_di = cmp_di.get_performance(method="pooled_with_average")
                counts = cmp.count_units_categories()
                counts_di = cmp.count_units_categories()

                new_data = {
                    "probe": probe,
                    "session": session_name,
                    "num_units": len(sorting.unit_ids),
                    "filter_option": filter_option,
                    "deepinterpolated": False,
                }
                new_data_di = new_data.copy()
                new_data_di["deepinterpolated"] = True
                new_data_di["num_units"] = len(sorting_di.unit_ids),

                new_data.update(perf_avg.to_dict())
                new_data.update(counts.to_dict())

                new_data_di.update(perf_avg_di.to_dict())
                new_data_di.update(counts_di.to_dict())

                new_df = pd.DataFrame([new_data])
                new_df_di = pd.DataFrame([new_data_di])
                new_df_session = pd.concat([new_df, new_df_di], ignore_index=True)

                if session_level_results is None:
                    session_level_results = new_df_session
                else:
                    session_level_results = pd.concat([session_level_results, new_df_session], ignore_index=True)

                # by unit
                perf_by_unit = cmp.get_performance(method="by_unit")
                perf_by_unit.loc[:, "probe"] = [probe] * len(perf_by_unit)
                perf_by_unit.loc[:, "session"] = [session_name] * len(perf_by_unit)
                perf_by_unit.loc[:, "filter_option"] = [filter_option] * len(perf_by_unit)
                perf_by_unit.loc[:, "deepinterpolated"] = [False] * len(perf_by_unit)

                perf_by_unit_di = cmp_di.get_performance(method="by_unit")
                perf_by_unit_di.loc[:, "probe"] = [probe] * len(perf_by_unit_di)
                perf_by_unit_di.loc[:, "session"] = [session_name] * len(perf_by_unit_di)
                perf_by_unit_di.loc[:, "filter_option"] = [filter_option] * len(perf_by_unit_di)
                perf_by_unit_di.loc[:, "deepinterpolated"] = [True] * len(perf_by_unit_di)

                new_unit_df = pd.concat([perf_by_unit, perf_by_unit_di], ignore_index=True)

                if unit_level_results is None:
                    unit_level_results = new_unit_df
                else:
                    unit_level_results = pd.concat([unit_level_results, new_unit_df], ignore_index=True)

            session_level_results.to_csv(results_folder / f"{dataset_name}-{session_name}-sessions.csv", index=False)
            unit_level_results.to_csv(results_folder / f"{dataset_name}-{session_name}-units.csv", index=False)
