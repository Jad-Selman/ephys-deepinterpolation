import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


#### IMPORTS #######
import sys
import json
import shutil
from pathlib import Path
import pandas as pd

# SpikeInterface
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.curation as scur
import spikeinterface.comparison as sc
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm


base_path = Path("..")

##### DEFINE DATASETS AND FOLDERS #######
from sessions import all_sessions_exp as all_sessions

n_jobs = 16

job_kwargs = dict(n_jobs=n_jobs, progress_bar=False, chunk_duration="1s")

data_folder = base_path / "data"
scratch_folder = base_path / "scratch"
results_folder = base_path / "results"


# Define training and testing constants
FILTER_OPTIONS = ["bp", "hp"]  # "hp", "bp", "no"


sorter_name = "pykilosort"
singularity_image = False
match_score = 0.7

sparsity_kwargs = dict(
    method="radius",
    radius_um=200,
)

# skip NN because extremely slow
qm_metric_names = [
    "num_spikes",
    "firing_rate",
    "presence_ratio",
    "snr",
    "isi_violation",
    "rp_violation",
    "sliding_rp_violation",
    "amplitude_cutoff",
    "drift",
    "isolation_distance",
    "l_ratio",
    "d_prime",
]


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "true":
            DEBUG = True
            OVERWRITE = True
        else:
            DEBUG = False

    session_dict = all_sessions
    filter_options = FILTER_OPTIONS

    json_files = [p for p in data_folder.iterdir() if p.name.endswith(".json")]
    if len(json_files) == 1:
        print(f"Found {len(json_files)} JSON config")
        session_dict = {}
        # each json file contains a session to run
        json_file = json_files[0]
        with open(json_file, "r") as f:
            config = json.load(f)
            probe = config["probe"]
            if probe not in session_dict:
                session_dict[probe] = []
            session = config["session"]
            assert (
                session in all_sessions[probe]
            ), f"{session} is not a valid session. Valid sessions for {probe} are:\n{all_sessions[probe]}"
            session_dict[probe].append(session)
            if "filter_option" in config:
                filter_options = [config["filter_option"]]
            else:
                filter_options = FILTER_OPTIONS
    elif len(json_files) > 1:
        print("Only 1 JSON config file allowed, using default sessions")

    print(f"Sessions:\n{session_dict}")
    print(f"Filter options:\n{filter_options}")

    si.set_global_job_kwargs(**job_kwargs)

    #### START ####
    probe_processed_folders = [p for p in data_folder.iterdir() if "processed_" in p.name and p.is_dir()]

    if len(probe_processed_folders) > 0:
        processed_folder = data_folder
    else:
        data_processed_subfolders = []
        for p in data_folder.iterdir():
            if p.is_dir() and len([pp for pp in p.iterdir() if "processed_" in pp.name and pp.is_dir()]) > 0:
                data_processed_subfolders.append(p)
        processed_folder = data_processed_subfolders[0]

    for probe, sessions in session_dict.items():
        print(f"Dataset {probe}")
        for session in sessions:
            print(f"\nAnalyzing session {session}\n")
            dataset_name, session_name = session.split("/")

            session_level_results_columns = [
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
            session_level_results = None

            unit_level_results_columns = [
                "dataset",
                "session",
                "probe",
                "filter_option",
                "unit_id",
                "deepinterpolated",
            ]
            unit_level_results = None

            matched_unit_level_results_columns = [
                "dataset",
                "session",
                "probe",
                "filter_option",
                "unit_id",
                "unit_id_di",
                "agreement_score",
            ]
            matched_unit_level_results = None

            for filter_option in filter_options:
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
                    try:
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
                    except:
                        print(f"Error sorting {session} with {sorter_name} and {filter_option}")
                        sorting = None

                if (sorting_output_folder / "sorting_di").is_dir() and not OVERWRITE:
                    print("\t\tLoading DI sorting")
                    sorting_di = si.load_extractor(sorting_output_folder / "sorting_di")
                else:
                    print(f"\t\tSpike sorting DI with {sorter_name}")
                    try:
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
                    except:
                        print(f"Error sorting DI {session} with {sorter_name} and {filter_option}")
                        sorting_di = None

                if sorting is not None and sorting_di is not None:
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
                else:
                    sorting_matched = None
                    sorting_di_matched = None

                new_row = {
                    "dataset": dataset_name,
                    "session": session_name,
                    "filter_option": filter_option,
                    "probe": probe,
                    "num_units": len(sorting.unit_ids) if sorting is not None else 0,
                    "num_units_di": len(sorting_di.unit_ids) if sorting_di is not None else 0,
                    "num_match": len(sorting_matched.unit_ids) if sorting_matched is not None else 0,
                    "sorting_path": str((sorting_output_folder / "sorting").relative_to(results_folder))
                    if sorting is not None
                    else None,
                    "sorting_path_di": str((sorting_output_folder / "sorting_di_").relative_to(results_folder))
                    if sorting_di is not None
                    else None,
                }

                print(
                    f"\n\t\tNum units: {new_row['num_units']} - Num units DI: {new_row['num_units_di']} - Num match: {new_row['num_match']}"
                )

                if session_level_results is None:
                    session_level_results = pd.DataFrame(columns=session_level_results_columns)
                session_level_results = pd.concat([session_level_results, pd.DataFrame([new_row])], ignore_index=True)

                if sorting_matched is not None:
                    # waveforms for all units
                    waveforms_scratch_folder = (
                        scratch_folder / f"waveforms_all_{dataset_name}_{session_name}_{filter_option}"
                    )
                    waveforms_all_folder = (
                        results_folder / f"waveforms_all_{dataset_name}_{session_name}_{filter_option}"
                    )
                    waveforms_all_folder.mkdir(exist_ok=True, parents=True)

                    if (waveforms_all_folder / "waveforms").is_dir() and not OVERWRITE:
                        print("\t\tLoad NO DI waveforms all")
                        we_all = si.load_waveforms(waveforms_all_folder / "waveforms")
                    else:
                        print("\t\tCompute NO DI waveforms all")
                        if sorting.sampling_frequency != recording.sampling_frequency:
                            print("\t\tSetting sorting sampling frequency to match recording")
                            sorting._sampling_frequency = recording.sampling_frequency
                        # first full, then sparse
                        we_dense = si.extract_waveforms(
                            recording,
                            sorting,
                            folder=waveforms_scratch_folder / "waveforms_dense",
                            n_jobs=n_jobs,
                            overwrite=True,
                            max_spikes_per_unit=100,
                        )
                        sparsity = si.compute_sparsity(we_dense, **sparsity_kwargs)
                        we_all = si.extract_waveforms(
                            recording,
                            sorting,
                            folder=waveforms_all_folder / "waveforms_all",
                            n_jobs=n_jobs,
                            overwrite=True,
                            sparsity=sparsity,
                        )
                        # remove dense folder
                        shutil.rmtree(waveforms_scratch_folder / "waveforms_dense")

                        print("\t\tCompute NO DI spike amplitudes")
                        _ = spost.compute_spike_amplitudes(we_all)
                        print("\t\tCompute NO DI spike locations")
                        _ = spost.compute_spike_locations(we_all)
                        print("\t\tCompute NO DI PCA scores")
                        _ = spost.compute_principal_components(we_all)

                        # finally, template and quality metrics
                        print("\t\tCompute NO DI template metrics")
                        tm_all = spost.compute_template_metrics(we_all)
                        print("\t\tCompute NO DI metrics")
                        qm_all = sqm.compute_quality_metrics(we_all, n_jobs=1, metric_names=qm_metric_names)

                    if (waveforms_all_folder / "waveforms_di").is_dir() and not OVERWRITE:
                        print("\t\tLoad DI waveforms all")
                        we_all_di = si.load_waveforms(waveforms_all_folder / "waveforms_di")
                    else:
                        print("\t\tCompute DI waveforms all")
                        if sorting_di.sampling_frequency != recording.sampling_frequency:
                            print("\t\tSetting sorting DI sampling frequency to match recording")
                            sorting_di._sampling_frequency = recording.sampling_frequency
                        # first full, then sparse
                        we_dense_di = si.extract_waveforms(
                            recording_di,
                            sorting_di,
                            folder=waveforms_scratch_folder / "waveforms_dense_di",
                            n_jobs=n_jobs,
                            overwrite=True,
                            max_spikes_per_unit=100,
                        )
                        sparsity_di = si.compute_sparsity(we_dense_di, **sparsity_kwargs)
                        we_all_di = si.extract_waveforms(
                            recording_di,
                            sorting_di,
                            folder=waveforms_all_folder / "waveforms_all_di",
                            n_jobs=n_jobs,
                            overwrite=True,
                            sparsity=sparsity_di,
                        )
                        # remove dense folder
                        shutil.rmtree(waveforms_scratch_folder / "waveforms_dense_di")

                        print("\t\tCompute DI spike amplitudes")
                        _ = spost.compute_spike_amplitudes(we_all_di)
                        print("\t\tCompute DI spike locations")
                        _ = spost.compute_spike_locations(we_all_di)
                        print("\t\tCompute DI PCA scores")
                        _ = spost.compute_principal_components(we_all_di)

                        # finally, template and quality metrics
                        print("\t\tCompute DI template metrics")
                        tm_all_di = spost.compute_template_metrics(we_all_di)

                        print("\t\tCompute DI metrics")
                        qm_all_di = sqm.compute_quality_metrics(we_all_di, n_jobs=1, metric_names=qm_metric_names)

                    waveforms_matched_folder = (
                        scratch_folder / f"waveforms_matched_{dataset_name}_{session_name}_{filter_option}"
                    )
                    waveforms_matched_folder.mkdir(exist_ok=True, parents=True)

                    if (waveforms_matched_folder / "waveforms").is_dir() and not OVERWRITE:
                        print("\t\tLoad NO DI waveforms matched")
                        we_matched = si.load_waveforms(waveforms_matched_folder / "waveforms")
                        qm_matched = we_matched.load_extension("quality_metrics").get_data()
                        tm_matched = we_matched.load_extension("template_metrics").get_data()
                    else:
                        print("\t\tSelect NO DI waveforms matched")
                        we_matched = we_all.select_units(
                            unit_ids=matched_unit_ids, new_folder=waveforms_matched_folder / "waveforms"
                        )
                        qm_matched = we_matched.load_extension("quality_metrics").get_data()
                        tm_matched = we_matched.load_extension("template_metrics").get_data()

                    if (waveforms_matched_folder / "waveforms_di").is_dir() and not OVERWRITE:
                        print("\t\tLoad DI waveforms matched")
                        we_matched_di = si.load_waveforms(waveforms_matched_folder / "waveforms_di")
                        qm_matched_di = we_matched_di.load_extension("quality_metrics").get_data()
                        tm_matched_di = we_matched_di.load_extension("template_metrics").get_data()
                    else:
                        print("\t\tSelect DI waveforms matched")
                        we_matched_di = we_all_di.select_units(
                            unit_ids=matched_unit_ids_di, new_folder=waveforms_matched_folder / "waveforms_di"
                        )
                        qm_matched_di = we_matched_di.load_extension("quality_metrics").get_data()
                        tm_matched_di = we_matched_di.load_extension("template_metrics").get_data()

                    ## add entries to unit-level results
                    if unit_level_results is None:
                        for metric in qm_all.columns:
                            unit_level_results_columns.append(metric)
                        for metric in tm_all.columns:
                            unit_level_results_columns.append(metric)
                        unit_level_results = pd.DataFrame(columns=unit_level_results_columns)

                    new_rows = {
                        "dataset": [dataset_name] * len(qm_all),
                        "session": [session_name] * len(qm_all),
                        "probe": [probe] * len(qm_all),
                        "filter_option": [filter_option] * len(qm_all),
                        "unit_id": we_all.unit_ids,
                        "deepinterpolated": [False] * len(qm_all),
                    }
                    new_rows_di = {
                        "dataset": [dataset_name] * len(qm_all_di),
                        "session": [session_name] * len(qm_all_di),
                        "probe": [probe] * len(qm_all_di),
                        "filter_option": [filter_option] * len(qm_all_di),
                        "unit_id": we_all_di.unit_ids,
                        "deepinterpolated": [True] * len(qm_all_di),
                    }
                    for metric in qm_all.columns:
                        new_rows[metric] = qm_all[metric].values
                        new_rows_di[metric] = qm_all_di[metric].values
                    for metric in tm_all.columns:
                        new_rows[metric] = tm_all[metric].values
                        new_rows_di[metric] = tm_all_di[metric].values
                    # append new entries
                    unit_level_results = pd.concat(
                        [unit_level_results, pd.DataFrame(new_rows), pd.DataFrame(new_rows_di)], ignore_index=True
                    )

                    ## add entries to matched unit-level results
                    if matched_unit_level_results is None:
                        for metric in qm_matched.columns:
                            matched_unit_level_results_columns.append(metric)
                            matched_unit_level_results_columns.append(f"{metric}_di")
                        for metric in tm_matched.columns:
                            matched_unit_level_results_columns.append(metric)
                            matched_unit_level_results_columns.append(f"{metric}_di")
                        matched_unit_level_results = pd.DataFrame(columns=matched_unit_level_results)

                    new_matched_rows = {
                        "dataset": [dataset_name] * len(qm_matched),
                        "session": [session_name] * len(qm_matched),
                        "probe": [probe] * len(qm_matched),
                        "filter_option": [filter_option] * len(qm_matched),
                        "unit_id": we_matched.unit_ids,
                        "unit_id_di": we_matched_di.unit_ids,
                    }

                    agreement_scores = []
                    for i in range(len(we_matched.unit_ids)):
                        agreement_scores.append(
                            comp.agreement_scores.at[we_matched.unit_ids[i], we_matched_di.unit_ids[i]]
                        )
                    new_matched_rows["agreement_score"] = agreement_scores
                    for metric in qm_matched.columns:
                        new_matched_rows[metric] = qm_matched[metric].values
                        new_matched_rows[f"{metric}_di"] = qm_matched_di[metric].values
                    for metric in tm_matched.columns:
                        new_matched_rows[metric] = tm_matched[metric].values
                        new_matched_rows[f"{metric}_di"] = tm_matched_di[metric].values
                    # append new entries
                    matched_unit_level_results = pd.concat(
                        [matched_unit_level_results, pd.DataFrame(new_matched_rows)], ignore_index=True
                    )

            if session_level_results is not None:
                session_level_results.to_csv(
                    results_folder / f"{dataset_name}-{session_name}-sessions.csv", index=False
                )
            if unit_level_results is not None:
                unit_level_results.to_csv(results_folder / f"{dataset_name}-{session_name}-units.csv", index=False)
            if matched_unit_level_results is not None:
                matched_unit_level_results.to_csv(
                    results_folder / f"{dataset_name}-{session_name}-matched-units.csv", index=False
                )
