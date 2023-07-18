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


base_path = Path("../../..")


data_folder = base_path / "data"
scratch_folder = base_path / "scratch"
results_folder = base_path / "results"


if __name__ == "__main__":

    # concatenate dataframes
    df_session = None
    df_units = None

    if (data_folder / "sortings").is_dir():
        data_base_folder = data_folder
    else:
        data_subfolders = [p for p in data_folder.iterdir() if (p / "sortings").is_dir()]
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

    # copy sortings to results folder

    # save concatenated dataframes
    df_session.to_csv(results_folder / "sessions.csv", index=False)
    df_units.to_csv(results_folder / "units.csv", index=False)
