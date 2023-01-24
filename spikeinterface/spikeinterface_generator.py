import tempfile
import json
import numpy as np

import spikeinterface.preprocessing as spre

from deepinterpolation.generator_collection import SequentialGenerator


class SpikeInterfaceGenerator(SequentialGenerator):
    """This generator is used when dealing with a SpikeInterface recording.
    The desired shape controls the reshaping of the input data before convolutions."""

    def __init__(self, recording, pre_frame=30, post_frame=30, pre_post_omission=1, desired_shape=(192, 2),
                 batch_size=100, steps_per_epoch=10, zscore=True, start_frame=None, end_frame=None):
        "Initialization"
        
        if zscore:
            recording_z = spre.zscore(recording)
        else:
            recording_z = recording

        self.recording = recording_z
        self.total_samples = recording.get_num_samples()
        assert len(desired_shape) == 2, "desired_shape should be 2D"
        assert desired_shape[0] * desired_shape [1] == recording.get_num_channels(), \
            f"The product of desired_shape dimensions should be the number of channels: {recording.get_num_channels()}"
        self.desired_shape = desired_shape
        
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self.total_samples
        
        assert end_frame > start_frame, "end_frame must be greater than start_frame"
        
        sequential_generator_params = dict()
        sequential_generator_params["steps_per_epoch"] = steps_per_epoch
        sequential_generator_params["pre_frame"] = pre_frame
        sequential_generator_params["post_frame"] = post_frame
        sequential_generator_params["batch_size"] = batch_size
        sequential_generator_params["start_frame"] = start_frame
        sequential_generator_params["end_frame"] = end_frame
        sequential_generator_params["total_samples"] = self.total_samples
        sequential_generator_params["pre_post_omission"] = pre_post_omission

        json_path = tempfile.mktemp(suffix=".json")
        with open(json_path, "w") as f:
            json.dump(sequential_generator_params, f)
        super().__init__(json_path)
        self._update_end_frame(self.total_samples)
        self._calculate_list_samples(self.total_samples)


    def __getitem__(self, index):
        # This is to ensure we are going through
        # the entire data when steps_per_epoch<self.__len__
        shuffle_indexes = self.generate_batch_indexes(index)

        input_full = np.zeros(
            [self.batch_size, self.desired_shape[0], self.desired_shape[1],
             self.pre_frame + self.post_frame],
            dtype="float32",
        )
        output_full = np.zeros(
            [self.batch_size, self.desired_shape[0], self.desired_shape[1], 1], dtype="float32"
        )

        for batch_index, frame_index in enumerate(shuffle_indexes):
            X, Y = self.__data_generation__(frame_index)

            input_full[batch_index, :, :, :] = X
            output_full[batch_index, :, :, :] = Y

        return input_full, output_full


    def __data_generation__(self, index_frame):
        "Generates data containing batch_size samples"

        # We reorganize to follow true geometry of probe for convolution
        input_full = np.zeros(
            [1, self.desired_shape[0], self.desired_shape[1],
             self.pre_frame + self.post_frame], dtype="float32"
        )
        output_full = np.zeros([1, self.desired_shape[0], self.desired_shape[1], 1], dtype="float32")

        start_frame = index_frame - self.pre_frame - self.pre_post_omission
        end_frame = index_frame + self.post_frame + self.pre_post_omission + 1
        full_traces = self.recording.get_traces(start_frame=start_frame, end_frame=end_frame).astype("float32")
        
        if full_traces.shape[0] == 0:
            print(f"Error! {index_frame}-{start_frame}-{end_frame}", flush=True)
        output_frame_index = self.pre_frame + self.pre_post_omission
        mask = np.ones(len(full_traces), dtype=bool)
        mask = np.ones(len(full_traces), dtype=bool)
        mask[output_frame_index - 1:output_frame_index + 2] = False    

        data_img_input = full_traces[mask]
        data_img_output = full_traces[output_frame_index][np.newaxis, :]

        # make 3d based on desired shape
        data_input_3d = data_img_input.reshape((-1, self.desired_shape[0], self.desired_shape[1]))
        data_output_3d = data_img_output.reshape((-1, self.desired_shape[0], self.desired_shape[1]))

        input_full[0] = data_input_3d.swapaxes(0, 1).swapaxes(1, 2)
        output_full[0] = data_output_3d.swapaxes(0, 1).swapaxes(1, 2)

        return input_full, output_full


    def reshape_output(self, output):
        return output.squeeze().reshape(-1, self.recording.get_num_channels())    
