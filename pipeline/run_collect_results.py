import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


#### IMPORTS #######
import shutil


from pathlib import Path
import pandas as pd


base_path = Path("..").resolve()

data_folder = base_path / "data"
scratch_folder = base_path / "scratch"
results_folder = base_path / "results"


if __name__ == "__main__":

    # concatenate dataframes
    df_session = None
    df_units = None

    probe_sortings_folders = [p for p in data_folder.iterdir() if "sortings_" in p.name and p.is_dir()]

    if len(probe_sortings_folders) > 0:
        data_base_folder = data_folder
    else:
        data_subfolders = [p for p in data_folder.iterdir() if p.is_dir()]
        data_base_folder = data_subfolders[0]

    session_csvs = [p for p in data_base_folder.iterdir() if "session" in p.name and p.suffix == ".csv"]
    unit_csvs = [p for p in data_base_folder.iterdir() if "unit" in p.name and p.suffix == ".csv"]

    for session_csv in session_csvs:
        if df_session is None:
            df_session = pd.read_csv(session_csv)
        else:
            df_session = pd.concat([df_session, pd.read_csv(session_csv)])

    for unit_csv in unit_csvs:
        if df_units is None:
            df_units = pd.read_csv(unit_csv)
        else:
            df_units = pd.concat([df_units, pd.read_csv(unit_csv)])

    # save concatenated dataframes
    df_session.to_csv(results_folder / "sessions.csv", index=False)
    df_units.to_csv(results_folder / "units.csv", index=False)

    # copy sortings to results folder
    sortings_folders = [p for p in data_base_folder.iterdir() if "sortings_" in p.name and p.is_dir()]
    sortings_output_base_folder = results_folder / "sortings"
    sortings_output_base_folder.mkdir(exist_ok=True)

    for sorting_folder in sortings_folders:
        sorting_folder_split = sorting_folder.name.split("_")
        dataset_name = sorting_folder_split[1]
        session_name = "_".join(sorting_folder_split[2:-1])
        filter_option = sorting_folder_split[-1]
        sorting_output_folder = sortings_output_base_folder / dataset_name / session_name / filter_option
        sorting_output_folder.mkdir(exist_ok=True, parents=True)
        for sorting_subfolder in sorting_folder.iterdir():
            shutil.copytree(sorting_subfolder, sorting_output_folder / sorting_subfolder.name)
