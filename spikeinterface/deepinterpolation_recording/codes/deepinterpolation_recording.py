import numpy as np
import os

import numpy as np
import os

import numpy as np
import os

from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
import numpy as np


__version__ = "0.1.0"


def has_tf(use_gpu=True, disable_tf_logger=True, memory_gpu=None):
    try:
        import_tf(use_gpu, disable_tf_logger, memory_gpu)
        return True
    except ImportError:
        return False
    
def import_tf(use_gpu=True, disable_tf_logger=True, memory_gpu=None):
    import tensorflow as tf

    if not use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if disable_tf_logger:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')

    tf.compat.v1.disable_eager_execution()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if memory_gpu is None:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    print("Setting memory growth")
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        else:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_gpu)])
    return tf



class DeepInterpolatedRecording(BasePreprocessor):
    name = 'deepinterpolate'

    def __init__(self, recording, model_path: str,
                 pre_frames: int = 30, post_frames: int = 30, pre_post_omission: int = 1,
                 batch_size=128, use_gpu: bool = True, disable_tf_logger: bool = True,
                 memory_gpu=None):
        assert has_tf(
            use_gpu, disable_tf_logger, memory_gpu), "To use DeepInterpolation, you first need to install `tensorflow`."
        
        self.tf = import_tf(use_gpu, disable_tf_logger, memory_gpu=memory_gpu)
        
        # try move model load here with spawn
        BasePreprocessor.__init__(self, recording)

        # first time retrieving traces check that dimensions are ok
        self.tf.keras.backend.clear_session()
        model = self.tf.keras.models.load_model(filepath=model_path)

        # check shape (this will need to be done at inference)
        network_input_shape = model.get_config()["layers"][0]["config"]["batch_input_shape"]
        desired_shape = network_input_shape[1:3]
        assert desired_shape[0]*desired_shape[1] == recording.get_num_channels(), "text"
        assert network_input_shape[-1] == pre_frames + post_frames


        self.model = model
        # add segment
        for segment in recording._recording_segments:
            recording_segment = DeepInterpolatedRecordingSegment(segment, self.model,
                                                                 pre_frames, post_frames, pre_post_omission,
                                                                 desired_shape, batch_size, use_gpu,
                                                                 disable_tf_logger, memory_gpu)
            self.add_recording_segment(recording_segment)

        self._preferred_mp_context = "spawn"
        self._kwargs = dict(recording=recording.to_dict(), model_path=model_path,
                            pre_frames=pre_frames, post_frames=post_frames, pre_post_omission=pre_post_omission,
                            batch_size=batch_size, use_gpu=use_gpu, disable_tf_logger=disable_tf_logger,
                            memory_gpu=memory_gpu)
        self.extra_requirements.extend(['tensorflow'])


class DeepInterpolatedRecordingSegment(BasePreprocessorSegment):

    def __init__(self, recording_segment, model,
                 pre_frames, post_frames, pre_post_omission,
                 desired_shape, batch_size, use_gpu, disable_tf_logger, memory_gpu):
        from spikeinterface_generator import SpikeInterfaceRecordingSegmentGenerator

        BasePreprocessorSegment.__init__(self, recording_segment)
        
        self.model = model
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.pre_post_omission = pre_post_omission
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.desired_shape=desired_shape

        # creating class dynamically to use the imported TF with GPU enabled/disabled based on the use_gpu flag
        self.SpikeInterfaceGenerator = SpikeInterfaceRecordingSegmentGenerator #define_input_generator_class( use_gpu, disable_tf_logger)

    def get_traces(self, start_frame, end_frame, channel_indices):
        n_frames = self.parent_recording_segment.get_num_samples()

        if start_frame == None:
            start_frame = 0

        if end_frame == None:
            end_frame = n_frames

        # for frames that lack full training data (i.e. pre and post frames including omissinos),
        # just return uninterpolated
        if start_frame < self.pre_frames+self.pre_post_omission:
            true_start_frame = self.pre_frames+self.pre_post_omission
            array_to_append_front = self.parent_recording_segment.get_traces(start_frame=0,
                                                                             end_frame=true_start_frame,
                                                                             channel_indices=channel_indices)
        else:
            true_start_frame = start_frame

        if end_frame > n_frames-self.post_frames-self.pre_post_omission:
            true_end_frame = n_frames-self.post_frames-self.pre_post_omission
            array_to_append_back = self.parent_recording_segment.get_traces(start_frame=true_end_frame,
                                                                            end_frame=n_frames,
                                                                            channel_indices=channel_indices)
        else:
            true_end_frame = end_frame

        # instantiate an input generator that can be passed directly to model.predict
        input_generator = self.SpikeInterfaceGenerator(recording_segment=self.parent_recording_segment,
                                                       start_frame=true_start_frame,
                                                       end_frame=true_end_frame,
                                                       pre_frame=self.pre_frames,
                                                       post_frame=self.post_frames,
                                                       pre_post_omission=self.pre_post_omission,
                                                       batch_size=self.batch_size)
        input_generator.randomize = False
        input_generator._calculate_list_samples(input_generator.total_samples)
        di_output = self.model.predict(input_generator, verbose=2)

        out_traces = input_generator.reshape_output(di_output)

        if true_start_frame != start_frame: # related to the restriction to be applied from the start and end frames around 0 and end
            out_traces = np.concatenate(
                (array_to_append_front, out_traces), axis=0)

        if true_end_frame != end_frame:
            out_traces = np.concatenate(
                (out_traces, array_to_append_back), axis=0)

        return out_traces[:, channel_indices]
