# This notebook serves as an example for the application of the inference on the dataset using the deepinterpolation-generated model.
# The data path is adjusted in a similar way as in the original repository by Lecoq et al. (2021), Allen Institute, Seattle, WA, USA, which can be accessed through the below link:
# https://github.com/AllenInstitute/deepinterpolation 
# It is necessary to adjust the directory of the input data accordingly, as well as the folder of the output folder location.

import os
from deepinterpolation.generic import JsonSaver, ClassLoader
import pathlib

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    generator_param = {}
    inferrence_param = {}

    # We are reusing the data generator for training here. Some parameters
    # like steps_per_epoch are irrelevant but currently needs to be provided
    generator_param["type"] = "generator"
    generator_param["name"] = "EphysGenerator"
    generator_param["pre_post_frame"] = 30
    generator_param["pre_post_omission"] = 1
    generator_param[
        "steps_per_epoch"
    ] = -1
    # No steps necessary for inference as epochs are not relevant.
    # -1 deactivate it.

    generator_param["train_path"] = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "..",
        "sample_data",
        "ephys_tiny_continuous.dat2",
    )

    generator_param["batch_size"] = 100
    generator_param["start_frame"] = 0
    generator_param["end_frame"] = 8000  # -1 to go until the end.
    generator_param[
        "randomize"
    ] = 0
    # This is important to keep the order and avoid the
    # randomization used during training

    inferrence_param["type"] = "inferrence"
    inferrence_param["name"] = "core_inferrence"

    # Replace this path to where you stored your model
    inferrence_param[
        "model_path"
    ] = "/Users/jadse/deepinterpolation/trials_different_batches/unet_single_ephys_1024_mean_absolute_error_2022_12_06_15_27_2022_12_06_15_27/2022_12_06_15_27_unet_single_ephys_1024_mean_absolute_error_2022_12_06_15_27_model.h5"

    # Replace this path to where you want to store your output file
    inferrence_param[
        "output_file"
    ] = "/Users/jadse/deepinterpolation/trials_different_batches/unet_single_ephys_1024_mean_absolute_error_2022_12_06_15_27_2022_12_06_15_27/ephys_tiny_continuous_deep_interpolation.h5"

    jobdir = "/Users/jadse/deepinterpolation/trials_different_batches/unet_single_ephys_1024_mean_absolute_error_2022_12_06_15_27_2022_12_06_15_27/"

    try:
        os.mkdir(jobdir)
    except Exception:
        print("folder already exists")

    path_generator = os.path.join(jobdir, "generator.json")
    json_obj = JsonSaver(generator_param)
    json_obj.save_json(path_generator)

    path_infer = os.path.join(jobdir, "inferrence.json")
    json_obj = JsonSaver(inferrence_param)
    json_obj.save_json(path_infer)

    generator_obj = ClassLoader(path_generator)
    data_generator = generator_obj.find_and_build()(path_generator)

    inferrence_obj = ClassLoader(path_infer)
    inferrence_class = inferrence_obj.find_and_build()(path_infer,
                                                       data_generator)

    # Except this to be slow on a laptop without GPU. Inference needs
    # parallelization to be effective.
    inferrence_class.run()