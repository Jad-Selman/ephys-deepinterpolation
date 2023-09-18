from pathlib import Path
import json

all_sessions_exp = {
    "NP1": [
        "aind-np1/625749_2022-08-03_15-15-06_ProbeA",
        "aind-np1/634568_2022-08-05_15-59-46_ProbeA",
        "aind-np1/634569_2022-08-09_16-14-38_ProbeA",
        "aind-np1/634571_2022-08-04_14-27-05_ProbeA",
        "ibl-np1/CSHZAD026_2020-09-04_probe00",
        "ibl-np1/CSHZAD029_2020-09-09_probe00",
        "ibl-np1/SWC054_2020-10-05_probe00",
        "ibl-np1/SWC054_2020-10-05_probe01",
    ],
    "NP2": [
        "aind-np2/595262_2022-02-21_15-18-07_ProbeA",
        "aind-np2/602454_2022-03-22_16-30-03_ProbeB",
        "aind-np2/612962_2022-04-13_19-18-04_ProbeB",
        "aind-np2/612962_2022-04-14_17-17-10_ProbeC",
        "aind-np2/618197_2022-06-21_14-08-06_ProbeC",
        "aind-np2/618318_2022-04-13_14-59-07_ProbeB",
        "aind-np2/618384_2022-04-14_15-11-00_ProbeB",
        "aind-np2/621362_2022-07-14_11-19-36_ProbeA",
    ],
}

all_sessions_sim = {
    "NP1": [
        "NP1/recording-0.h5",
        "NP1/recording-1.h5",
        "NP1/recording-2.h5",
        "NP1/recording-3.h5",
        "NP1/recording-4.h5",
    ],
    "NP2": [
        "NP2/recording-0.h5",
        "NP2/recording-1.h5",
        "NP2/recording-2.h5",
        "NP2/recording-3.h5",
        "NP2/recording-4.h5",
    ],
}

FILTER_OPTIONS = ["hp", "bp"]


def generate_job_config_list(output_folder, split_probes=True, split_filters=True, dataset="exp"):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    if dataset == "exp":
        all_sessions = all_sessions_exp
    else:
        all_sessions = all_sessions_sim

    i = 0
    for probe, sessions in all_sessions.items():
        if split_probes:
            i = 0
            probe_folder = output_folder / probe
            probe_folder.mkdir(exist_ok=True)
        else:
            probe_folder = output_folder

        for session in sessions:
            d = dict(session=session, probe=probe)

            if split_filters:
                for filter_option in FILTER_OPTIONS:
                    d["filter_options"] = filter_option
                    with open(probe_folder / f"job{i}.json", "w") as f:
                        json.dump(d, f)
                    i += 1
            else:
                with open(probe_folder / f"job{i}.json", "w") as f:
                    json.dump(d, f)
                i += 1

