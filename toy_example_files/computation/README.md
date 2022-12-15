In the computational aspect, the uploaded files (example_tiny_ephys_training.py and example_tiny_ephys_inference.py) are the only ones used for the generation of the results.

example_tine_ephys_training.py:
-------------------------------

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


example_tiny_ephys_inference.py:
--------------------------------
the path of the ".h5", generated in the above step, is inserted at the specific code line. 
The samples to be taken into consideration for the inference process are specified by adjusting the start and end frame. The user can as well change the number of mini batches taken into consideration (here, not altered).
After running the code file, a ".h5" file, holding the resulting inferred data, is generated, and which can visualized using the plotting code files in the "visualization" folder.
