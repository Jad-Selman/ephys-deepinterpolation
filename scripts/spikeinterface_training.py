import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

from tensorflow import keras
import tensorflow as tf

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

from deepinterpolation.trainor_collection import core_trainer
from deepinterpolation.network_collection import unet_single_ephys_1024
from deepinterpolation.generic import ClassLoader

# make python functions and classes in "src" available here
import sys
sys.path.append("../src")

# the generator is in the "spikeinterface_generator.py"
from spikeinterface_generator import SpikeInterfaceGenerator
from deepinterpolation_recording import DeepInterpolatedRecording

print(tf.config.list_physical_devices('GPU'))

FULL_INFERENCE = False

# Define training and testing constants (@Jad you can gradually increase this)
TRAINING_START_S = 0
TRAINING_END_S = 2
TESTING_START_S = 70
TESTING_END_S = 70.1
DESIRED_SHAPE = (192, 2)

FILTER_OPTIONS = ["bp", "hp", "no"] # "hp", "bp"

external_data_folder = Path("/home/buccino/data/")
data_folder = Path("../data")

pre_frame = 30
post_frame = 30
pre_post_omission = 1
n_jobs = 16


# if __name__ == "__main__":

### Load NP2 dataset (and preprocess)

# example of data generation in spike interface
folder_path = data_folder / "Neuropixels2.0_Recording/open-ephys-np2/595262_2022-02-22_16-47-26/"
recording = se.read_openephys(folder_path)

for FILTER in FILTER_OPTIONS:

    training_time = TRAINING_END_S - TRAINING_START_S
    testing_time = TESTING_END_S - TESTING_START_S
    trained_model_folder = data_folder / f"di_{FILTER}_filter_t{training_time}s_v{testing_time}s"
    trained_model_folder.mkdir(exist_ok=True)

    recording_folder = data_folder / "recording_saved"
    if recording_folder.is_dir():
        shutil.rmtree(recording_folder)

    assert FILTER in ("no", "hp", "bp"), "Wrong filter option!"
    if FILTER == "hp":
        rec_f = spre.highpass_filter(recording)
        rec_f = rec_f.save(folder=recording_folder, n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
    elif FILTER == "bp":
        rec_f = spre.bandpass_filter(recording)
        rec_f = rec_f.save(folder=recording_folder, n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)
    else:
        rec_f = recording

    rec_norm = spre.zscore(rec_f)


    ### Test SpikeInterfaceGenerator behavior

    si_generator = SpikeInterfaceGenerator(rec_norm, batch_size=10, zscore=False)
    input_0, output_0 = si_generator[0]
    si_generator.batch_size

    print(f"Input shape: {input_0.shape}")
    print(f"Output shape: {output_0.shape}")


    ### Perform small training

    start_frame_training = int(TRAINING_START_S * rec_norm.sampling_frequency)
    end_frame_training = int(TRAINING_END_S * rec_norm.sampling_frequency)
    start_frame_test = int(TESTING_START_S * rec_norm.sampling_frequency) 
    end_frame_test = int(TESTING_END_S * rec_norm.sampling_frequency)

    # Training (from core_trainor class)
    training_data_generator = SpikeInterfaceGenerator(rec_norm, zscore=False, 
                                                    pre_frame=pre_frame, post_frame=post_frame,
                                                    pre_post_omission=pre_post_omission,
                                                    start_frame=start_frame_training,
                                                    end_frame=end_frame_training,
                                                    desired_shape=DESIRED_SHAPE)
    test_data_generator = SpikeInterfaceGenerator(rec_norm, zscore=False,
                                                pre_frame=pre_frame, post_frame=post_frame,
                                                pre_post_omission=pre_post_omission,
                                                start_frame=start_frame_test,
                                                end_frame=end_frame_test,
                                                steps_per_epoch=-1,
                                                desired_shape=DESIRED_SHAPE)


    # Those are parameters used for the network topology
    network_params = dict()
    network_params["type"] = "network"
    # Name of network topology in the collection
    network_params["name"] = "unet_single_ephys_1024"

    network_json_path = trained_model_folder / "network_params.json"
    with open(network_json_path, "w") as f:
        json.dump(network_params, f)

    network_obj = ClassLoader(network_json_path)
    data_network = network_obj.find_and_build()(network_json_path)

    training_params = dict()
    training_params["loss"] = "mean_absolute_error"

    training_params["model_string"] = f"{network_params['name']}_{training_params['loss']}"
    training_params["output_dir"] = str(trained_model_folder)
    # We pass on the uid
    training_params["run_uid"] = "first_test"

    # We convert to old schema
    training_params["nb_gpus"] = 1
    training_params["type"] = "trainer"
    training_params["steps_per_epoch"] = 10
    training_params["period_save"] = 100
    training_params["apply_learning_decay"] = 0
    training_params["nb_times_through_data"] = 1
    training_params["learning_rate"] = 0.0001
    training_params["pre_post_frame"] = 1
    training_params["loss"] = "mean_absolute_error"
    training_params["nb_workers"] = 2
    training_params["caching_validation"] = False


    training_json_path = trained_model_folder / "training_params.json"
    with open(training_json_path, "w") as f:
        json.dump(training_params, f)


    training_class = core_trainer(
        training_data_generator, test_data_generator, data_network,
        training_json_path
    )

    print("created objects for training")
    training_class.run()

    print("training job finished - finalizing output model")
    training_class.finalize()

    ### Test inference

    # Re-load model from output folder
    model_path = trained_model_folder / f"{training_params['run_uid']}_{training_params['model_string']}_model.h5"
    
    rec_di = rec = DeepInterpolatedRecording(rec_norm, model_path=model_path, pre_frames=pre_frame, 
                                             post_frames=post_frame, pre_post_omission=pre_post_omission, disable_tf_logger=True,
                                             use_gpu=True)
    
    if FULL_INFERENCE:
        rec_di_saved = rec_di.save(folder=data_folder/f"inference_{FILTER}_filter_t{training_time}s_v{testing_time}s", n_jobs=8)
