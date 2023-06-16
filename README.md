# Project: Investigation and Extension of DeepInterpolation to Denoise High-density Electrophysiology Recordings


## Project Description
---------------------

The primary goal of this project is to make DeepInterpolation (DI)-based denoising (Lecoq et al.,2021) more flexible and readily usable by the electrophysiology community. In order to do so, we plan to integrate the DI
pipeline into the SpikeInterface project (Buccino et al. 2020). 

SpikeInterface is a Python package that aims to unify the analysis of extracellular electrophysiology. It can read data from tens of proprietary formats and provides functions for pre- and post-processing, visualization,
quality metrics, and spike sorting comparison.

Currently, deepinterpolation is available as a preprocessing module in SpikeInterface and it
enables the user to perform the inference part of DeepInterpolation directly from SpikeInterface.
Of course, this step requires the user to have access to a pre-trained model.

The objectives can be listed as following:

- Aim 1
    The first objective of the project is to integrate the DeepInterpolation Training module into the
    SpikeInterface API. In doing so, the software will need to adjust the architecture and parameters
    of the neural network depending on the characteristics of the input recording(s) (e.g., probe
    geometry, sampling frequency). The integration of the training phase would also facilitate the
    fine-tuning of an existing model: starting from a pre-trained model, a user could run a few
    batches of training to tune the model to a specific dataset and improve its performance.

- Aim 2
    The second objective is to train and publicly share pre-trained models for different kinds of datasets.
    In particular, the plan is to build DeepInterpolation models for Neuropixels 2.0 probes (Steinmetz
    et al. 2021), that will soon become the standard in large-scale electrophysiology research. In
    order to share and document these valuable trained models with the users, one possibility could
    be to upload them to an online platform such as HuggingFace. Hugging Face is a community
    built collection of pre-trained machine learning models, which also enables users to version tag
    and document the models.

- Aim 3
    The third part of the project aims at investigating and quantitatively assessing whether
    DeepInterpolation improves spike sorting performance. To do so, we plan to use several
    ground-truth recording strategies (including real ground-truth, hybrid data, and fully synthetic
    data (Buccino, Garcia, and Yger 2022)) to benchmark whether and to what extent processing the
    recordings with DeepInterpolation improves spike sorting performance. In addition, this step will
    also be important to check that DeepInterpolation does not inject artifacts that may hinder spike
    sorting quality.


To accomplish the aforementioned objectives, and to fulfill the aim of reproducibility and interoperability, the project is publicly published as a github repository entitled: ephys-deepinterpolation.


** More information about deepinterpolation can be found in the following cited paper:
    
    - Lecoq, J., Oliver, M., Siegle, J.H. et al. Removing independent noise in systems neuroscience data using      DeepInterpolation. Nat Methods 18, 1401–1408 (2021).
    https://doi.org/10.1038/s41592-021-01285-2

The github repository of Deepinterpolation [link](https://github.com/AllenInstitute/deepinterpolation)



** More information about spikeinterface can be found on the following cited paper:
    
    Alessio P Buccino, Cole L Hurwitz, Samuel Garcia, Jeremy Magland, Joshua H Siegle, Roger Hurwitz, Matthias H Hennig (2020) SpikeInterface, a unified framework for spike sorting eLife 9:e61834
    https://doi.org/10.7554/eLife.61834

  
The webpage of spikeinterface: [link](https://spikeinterface.readthedocs.io/en/latest/index.html)
The github repository of spikeinterface: [link](https://github.com/SpikeInterface/spikeinterface) 



## Repository Organization

The repository is organized in 3 main folders: `src` (source), `scripts` and `notebooks`.

- `src`: is the folder containing the source code "deepinterpolation_recording.py" and "spikeinterface_generator.py", both of which contains the main classes and functions that are the essential backbone for rebuilding input data recordings -essential for the generation of a trained model that reconstructs the input data according to a desired shape -, and for the application of adjusted deepinterpolation code to be applied on the input data. These classes and functions are called into the used scripts and notebooks.

- `scripts`: contaning python scripts (each discribed with an internal readme note).
  
- `notebooks`: containing jupyter notebooks that in addition to other outcomes, produce the same output as the scripts in "scripts" folder, but more commented and in step-by-step fashion.   

Note: it is important to keep in mind that the both the scripts and notebooks assume that the input data, as well as the generated models and output are added into a folder named "data", that is present within the same directory containing both "scripts" and "notebooks" folders. It is important for the user to create such folder that should be named "data".

## Installation

As this project relies on both deepinterpolation and spikeinterface, their dependencies must be installed.

Requirements for spikeinterface can be installed following the instructions present on this link:
[link] (https://github.com/SpikeInterface/spikeinterface/blob/main/installation_tips/full_spikeinterface_environment_linux_dandi.yml)

Requirements for deepinterpolation can be installed following the instructions present on this link:
[link] (https://github.com/AllenInstitute/deepinterpolation/blob/master/requirements.txt)

to be completed ...

## How to reproduce

to be completed ...

## Contributors

### Principal Investigator:
---------------------
Jad Selman - Politecnico di Milano

### Main supervisors:
-----------------
Alessio Buccino - Allen Institute for Neural Dynamics
Jerome Lecoq - Allen Institute for Brain Science

### Affiliated supervisor:
--------------------
Alessandra Pedrocchi - Politecnico di Milano