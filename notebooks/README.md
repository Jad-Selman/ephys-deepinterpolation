deepinterpolation_recording_zscores_upscaling.ipynb:
----------------------------------------------------
This notebook rescales the values that have been generated the inference phase by "gain" and "offset" in order to reverse the effect of the data normalization by zscoring. It also saves the file in "int16" format in order to make readily available to be used in spikesorting.

This notebook also allows visual comparison of the zscored vs. upsampled values, for reliability purposes, this is why it is initialy transformed in "float" data type.

This notebook provides two methods to save the upscaled values as "int16" data type. The first using "GPU", if present, and the other using "CPU". The speed of the data transformation depends on the chosen number of chunks and number of jobs, both of which the user can alter.



deepinterpolation_recording.ipynb:
----------------------------------
This notebook is responsible of:
- creating different deepinterpolated models of the recording after filtering and zscoring the input data in 3 different condition (bandpass then deepinterpolation, deepinterpolation then bandpassing, and highpass filtering then deepinterpolation). This section is already addressed and fully automated in the script "spikeinterface_training.py" present in the "script" folder, but here the steps are more explicitly presented and for each filtering approach the statistical directly annexed for clarity and educational purposes.

- conducting statistical analysis of the predicted recording with respect to the groundtruth data: visualizing groundtruth data, predicted data, and the difference between them, power spectrum density using welch method of the difference, covariance matrices, Kolmogoroc-Smirnov test of the differnce for whitness testing.
  


Example_ephys_deepinterpolation_training.ipynb:
-----------------------------------------------

To uplaoded the desired dataset, the path should be adjusted at the corresponding code line. 

The hyperparameters corresponding to the training phase includes the label "generator_param", while those corresponding to the validation phase includes the label "generator_test_param".

Both the training and validation hyperparameters can be altered in this file, and the main ones are:
	- start and end frame: separately specifying the frames of the dataset that should be included in the training and the validation phases, and thus the number of training and validation parameters taken into consideration.
	- size of minibatch: the number of data samples taken into consideration number of minibatches per epoch: number minibatches taken into consideration per epoch of training. 
	- steps per epoch: the minibatches are evenly grouped together and executed according to how many steps the user wishes the training parameters to be updated per epoch. 
	- nb times through data: the number of times the training phase is counting the same data samples in it for training purposes.

Note: the number of epochs is automatically calculated as: (number of training samples)/(size of minibatch)

Other parameters could be altered as well, but the aforementioned ones may have the strongest impact on the results.

Once the location and the hyperparameters are adjusted, the code file can be run and it automatically generates:
	- a ".h5" file holding the trained parameters.	
	- a figure describing the training and validation loss progress across the epochs.



Example_ephys_inference_application.ipynb:
--------------------------------
the path of the ".h5", generated in the above step, is inserted at the specific code line. 
The samples to be taken into consideration for the inference process are specified by adjusting the start and end frame. The user can as well change the number of mini batches taken into consideration (here, not altered).
After running the code file, a ".h5" file, holding the resulting inferred data, is generated, and which can visualized using the plotting code files in the "visualization" folder.
