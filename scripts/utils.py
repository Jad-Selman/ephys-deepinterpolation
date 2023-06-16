
import json
import sys
import shutil

import spikeinterface as si
import spikeinterface.preprocessing as spre

# DeepInterpolation
from deepinterpolation.trainor_collection import core_trainer
from deepinterpolation.network_collection import unet_single_ephys_1024
from deepinterpolation.generic import ClassLoader

# Import local classes for DI+SI
sys.path.append("../src")

# the generator is in the "spikeinterface_generator.py"
from spikeinterface_generator import SpikeInterfaceGenerator
from deepinterpolation_recording import DeepInterpolatedRecording




def train_di_model(recording, session, filter_option, train_start_s, train_end_s, 
                   test_start_s, test_end_s, data_folder, full_inference, model_name, di_kwargs,
                   overwrite=False, use_gpu=True):
    """_summary_

    Parameters
    ----------
    recording : _type_
        _description_
    session : _type_
        _description_
    filter_option : _type_
        _description_
    train_start_s : _type_
        _description_
    train_end_s : _type_
        _description_
    test_start_s : _type_
        _description_
    test_end_s : _type_
        _description_
    data_folder : _type_
        _description_
    full_inference : _type_
        _description_
    di_kwargs : _type_
        _description_
    """
    model_folder = data_folder / "models" / session
    model_folder.mkdir(exist_ok=True, parents=True)
    trained_model_folder = model_folder / model_name

    pre_frame = di_kwargs["pre_frame"]
    post_frame = di_kwargs["post_frame"]
    pre_post_omission = di_kwargs["pre_post_omission"]
    desired_shape = di_kwargs["desired_shape"]
    inference_n_jobs = di_kwargs["inference_n_jobs"]
    inference_chunk_duration = di_kwargs["inference_chunk_duration"]

    # pre-process
    assert filter_option in ("no", "hp", "bp"), "Wrong filter option!"
    if filter_option == "hp":
        rec_f = spre.highpass_filter(recording)
    elif filter_option == "bp":
        rec_f = spre.bandpass_filter(recording)
    else:
        rec_f = recording

    rec_processed = rec_f
    rec_norm = spre.zscore(rec_processed)

    ### Define params
    start_frame_training = int(train_start_s * rec_norm.sampling_frequency)
    end_frame_training = int(train_end_s * rec_norm.sampling_frequency)
    start_frame_test = int(test_start_s * rec_norm.sampling_frequency) 
    end_frame_test = int(test_end_s * rec_norm.sampling_frequency)

    # Those are parameters used for the network topology
    network_params = dict()
    network_params["type"] = "network"
    # Name of network topology in the collection
    network_params["name"] = "unet_single_ephys_1024"
    training_params = dict()
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
    training_params["model_string"] = f"{network_params['name']}_{training_params['loss']}"
    

    if not trained_model_folder.is_dir() or overwrite:
        trained_model_folder.mkdir(exist_ok=True)

        # Training (from core_trainor class)
        training_data_generator = SpikeInterfaceGenerator(rec_norm, zscore=False, 
                                                        pre_frame=pre_frame, post_frame=post_frame,
                                                        pre_post_omission=pre_post_omission,
                                                        start_frame=start_frame_training,
                                                        end_frame=end_frame_training,
                                                        desired_shape=desired_shape)
        test_data_generator = SpikeInterfaceGenerator(rec_norm, zscore=False,
                                                    pre_frame=pre_frame, post_frame=post_frame,
                                                    pre_post_omission=pre_post_omission,
                                                    start_frame=start_frame_test,
                                                    end_frame=end_frame_test,
                                                    steps_per_epoch=-1,
                                                    desired_shape=desired_shape)


        network_json_path = trained_model_folder / "network_params.json"
        with open(network_json_path, "w") as f:
            json.dump(network_params, f)

        network_obj = ClassLoader(network_json_path)
        data_network = network_obj.find_and_build()(network_json_path)

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
    else:
        print("Loading pre-trained model")

    ### Test inference

    # Re-load model from output folder
    model_path = trained_model_folder / f"{training_params['run_uid']}_{training_params['model_string']}_model.h5"
    
    rec_di = DeepInterpolatedRecording(rec_norm, model_path=model_path, pre_frames=pre_frame, 
                                       post_frames=post_frame, pre_post_omission=pre_post_omission, 
                                       disable_tf_logger=True,
                                       use_gpu=use_gpu)
    
    if full_inference:
        deepinterpolated_folder = data_folder / "deepinterpolated" / session
        deepinterpolated_folder.mkdir(exist_ok=True, parents=True)
        output_folder = deepinterpolated_folder / model_name
        if output_folder.is_dir() and overwrite:
            shutil.rmtree(output_folder)
            rec_di = DeepInterpolatedRecording(rec_norm, model_path=model_path, pre_frames=pre_frame, 
                                               post_frames=post_frame, pre_post_omission=pre_post_omission, 
                                               disable_tf_logger=True,
                                               use_gpu=use_gpu)
            rec_di_saved = rec_di.save(folder=output_folder, n_jobs=inference_n_jobs,
                                    chunk_duration=inference_chunk_duration)
        else:
            rec_di_saved = si.load_extractor(output_folder)
    else:
        rec_di_saved = rec_di

    # apply inverse z-scoring
    inverse_gains = 1 / rec_norm.gain
    inverse_offset = - rec_norm.offset * inverse_gains
    rec_di_inverse_zscore = spre.scale(rec_di_saved, gain=inverse_gains, offset=inverse_offset, dtype='float')
    return rec_processed, rec_di_inverse_zscore
