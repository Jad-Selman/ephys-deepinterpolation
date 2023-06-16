spikeinterface_training.py:
---------------------------
this script reads an input data recording file, generates a model that has reconstructed the original data configuration according to an applied filtering option, applies a filtering option (filtering is optional) to the input data recording then normalizes it (normalization is optional), applies deepinterpolation procedure (Lecoq et al., 2021), and finally generates a deepinterpolated version of the input data (inference phase). 

The user can either choose to apply a highpass or bandpass filtering process to the data, or no filtering at all. It is important to understand that the the same filtering approach should be adopted for both the "trained_model_folder" (model in which the original data distribution has been reconstructed) and the "rec_f" (filtered recording), both of which will be the main inputs in the deepinterpolation phase. This has been well addressed and hard coded in this script for facilitation and accessibility purposes.   

It should be noted that the data input path is relative with respect to a data folder named "data" present outside the folder "scripts", and within which the input recording is added. In case of additional changes or further alterations to the data directory by the user, they should reflect this change in the code as well. The same thing applies for the generated output.

With regard to the the generation of a model that reconstructs the input data, it is important to understant that the pre, post and omission frames are adusjted according to the Deepinterpolation approach (Lecoq et al., 2021), and thus they are to be left accordingly. Whereas for "TRAINING_START_S", "TRAINING_END_S", "TESTING_START_S", and "TESTING_END_S", they are hyperparamters in "seconds" time unit, and they are adjusted according to the user's willingness to train and validate the model according to input data time frame. It is important to keep in mind that this is a relatively long procedure, and depends on the machine's computational capabilities. As for "DESIRED_SHAPE" being (192, 2), this reflects that the output desired shape is in conformation with the "Neuropixels2.0" arrays' spatial configuration, and should only be altered to conform the arrays' spatial configuration of the measuring electrode used in the data acquisition process.


Example_ephys_deepinterpolation_training.py:
--------------------------------------------

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


Example_ephys_inference_application.py:
---------------------------------------
the path of the ".h5", generated in the above step, is inserted at the specific code line. 
The samples to be taken into consideration for the inference process are specified by adjusting the start and end frame. The user can as well change the number of mini batches taken into consideration (here, not altered).
After running the code file, a ".h5" file, holding the resulting inferred data, is generated, and which can visualized using the plotting code files in the "visualization" folder.


visualize_content_inference_file.py:
-----------------------------------
in order to generate a ".png" file image of the output of the inference, it is enough to move this script into the same folder containing the genrated output. The script reads the inference data output file being named "ephys_tiny_continuous_deep_interpolation.h5", the standard name used in in the original repository by Lecoq et al. (Allen Institute, 2021). In case the inference output is of a different name, it should be reflected in this script.
Kind reminder to change the titles and the ".png" file as the user desires.


visualize_content_rawdata_file.py:
----------------------------------
in order to generate a ".png" file image of the original raw data, it is enough to move this script into the same folder containing the data file. The script reads the data file file being named "ephys_tiny_continuous.dat2", the standard name used in in the original repository by Lecoq et al. (Allen Institute, 2021). In case the data file is of a different name, it should be reflected in this script.
Kind reminder to change the titles and the ".png" file as the user desires.
