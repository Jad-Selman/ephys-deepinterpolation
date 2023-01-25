import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from tensorflow import keras

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre

from deepinterpolation.trainor_collection import core_trainer
from deepinterpolation.network_collection import unet_single_ephys_1024
from deepinterpolation.generic import ClassLoader

# the generator is in the "spikeinterface_generator.py"
from spikeinterface_generator import SpikeInterfaceGenerator


# Define training and testing constants (@Jad you can gradually increase this)
TRAINING_START_S = 0
TRAINING_END_S = 10
TESTING_START_S = 100
TESTING_END_S = 101
DESIRED_SHAPE = (192, 2)


pre_frame = 30
post_frame = 30
pre_post_omission = 1


### Load NP2 dataset (and preprocess)

# example of data generation in spike interface
folder_path = "/home/alessio/Documents/data/allen/npix-open-ephys/595262_2022-02-22_16-47-26/"
recording = se.read_openephys(folder_path)

rec_f = spre.bandpass_filter(recording)
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
output_folder = Path("test_training")
output_folder.mkdir(exist_ok=True)

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

network_json_path = output_folder / "network_params.json"
with open(network_json_path, "w") as f:
    json.dump(network_params, f)

network_obj = ClassLoader(network_json_path)
data_network = network_obj.find_and_build()(network_json_path)

training_params = dict()
training_params["loss"] = "mean_absolute_error"

training_params["model_string"] = f"{network_params['name']}_{training_params['loss']}"
training_params["output_dir"] = str(output_folder)
# We pass on the uid
training_params["run_uid"] = "first_test"

# We convert to old schema
training_params["nb_gpus"] = 1
training_params["type"] = "trainer"
training_params["steps_per_epoch"] = 10
training_params["period_save"] = 5
training_params["apply_learning_decay"] = 0
training_params["nb_times_through_data"] = 1
training_params["learning_rate"] = 0.0001
training_params["pre_post_frame"] = 1
training_params["loss"] = "mean_absolute_error"
training_params["nb_workers"] = 2
training_params["caching_validation"] = False


training_json_path = output_folder / "training_params.json"
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
model_path = output_folder / f"{training_params['run_uid']}_{training_params['model_string']}_model.h5"
keras.backend.clear_session()
model = keras.models.load_model(filepath=model_path)

# check shape (this will need to be done at inference)
network_input_shape = model.get_config()["layers"][0]["config"]["batch_input_shape"]
assert network_input_shape[1:] == DESIRED_SHAPE + (pre_frame + post_frame,)

sample_input, original_data = test_data_generator[0]

output = training_class.local_model.predict(sample_input)
output_data = test_data_generator.reshape_output(output)
input_data = original_data.squeeze().reshape(-1, recording.get_num_channels())

fig, axs = plt.subplots(ncols=2)
axs[0].imshow(input_data.T, origin="lower", cmap="RdGy_r")
axs[1].imshow(output_data.T, origin="lower", cmap="RdGy_r")