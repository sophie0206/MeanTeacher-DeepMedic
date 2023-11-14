# 	Semi-supervised Brain Lesion Segmentation with an Adapted Mean Teacher Model,  [Paper](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_43) published at IPMI 2019
#### Official Implementation of the Mean Teacher method for brain lesion segmentation, as well as other semi-supervised learning methods for comparison
The code is using on [DeepMedic](https://github.com/deepmedic/deepmedic) as the backbone architecture. Please refer to the environment requirements in DeepMedic.

### Directory
DeepMedic consists of several methods, each in its own directory:

- `deepmedicMT`: Contains code for the Mean Teacher method.
- `deepmedicUDA`: Contains code for the Unsupervised Domain Adaptation method.
- `deepmedicEN`: Contains code for the Evaluation Network method.
- `deepmedicO`: Contains the original DeepMedic code.

Each folder contains an executable file named `deepMedicRun`, which is used to run the entire model. 


### Example:

To run the Mean Teacher method, use the following command:

`CUDA_VISIBLE_DEVICES=2 ./deepMedicRunMT
-model ./examples/configFiles/deepMedic/model/modelConfigMTStroke.cfg -train ./examples/configFiles/deepMedic/train/trainConfigMTStroke10.cfg -dev cuda2`

## Configuration Files
Configuration files are divided into three categories: model configurations, testing and training configurations.
- **Model Config Files**: 
  - These files define the structural parameters of the model.
  - Key parameters to note include `numberOfOutputClasses`, `numberOfInputChannels`, `batch size`, and `segmentsDimTrain` (size of training segments).

- **Training Config Files**: 
  - These files define parameters related to model training.
  - Important parameters include the number of epochs and subepochs, the number of segments loaded into the GPU at a time, and the corresponding files for input channels, ground truth labels, and ROIMasks. 
  - These corresponding files should contain the absolute paths of the respective images, set according to training needs.

In the corresponding `trainConfig` files for semi-supervised models, I have added three parameters to represent:
  - The training image paths for unlabeled data (`targetDomainChannelsTraining`).
  - The ground-truth image paths for unlabeled data (`TDgtLabelsTraining`). Note that GT-Labels for unlabeled data are used as masks for lesion area sampling and are not used for training.
  - The ROIMask image paths for unlabeled data (`DDroiMasksTraining`). For Stroke data, the ROIMask is the mask of the brain area, used for weighted-sampling. This involves extracting an equal number of segments from the foreground and background. The sampling method will be explained in detail in the code documentation.
    
## Data Preprocessing

Each set of training data for DeepMedic is stored in a separate folder. The folder includes NIfTI format images:
- Different modality images (as different input channels).
- Ground-Truth Label (background must be 0).
- WeightMap0 (mask for non-lesion areas).
- WeightMap1 (mask for lesion areas).
- BrainMask (mask for the brain area, excluding the skull).

These images are generated and saved to their respective storage paths using the Data Pre-processing code. It is crucial that all these images have the same shape.

Additionally, DeepMedic requires data normalization to a distribution with a mean of 0 and a variance of 1.

For more specific requirements of data preprocessing, please refer to the DeepMedic GitHub page.

## Code Tutorial

### Overview
DeepMedic is built using the TensorFlow deep learning framework, specifically version 1.7.0. 

### Training
- **trainSession.py**: Creates the computation graph and session, initializes model parameters. 
- **training.py**: Provides operations and `feeds_dict`, runs the session, executes operations, and returns results. The computation graph represents the entire architecture of the 3D CNN model. Relevant files include `cnn3d.py`, `pathways.py`, and `layers.py`. The construction of CNN in DeepMedic is divided into three levels: building different types of layers, constructing a pathway from multiple layers, and combining different types of pathways to form the final 3D CNN model.

- **Layers.py**: Contains three classes defining properties and operations for Convolutional Layer (`ConvLayer`), Fully Connected Layer (`FcLayer`), and Softmax Layer (`SoftmaxLayer`). The `makeLayer` function creates the entire convolutional layer and its parameters, with detailed operations defined in `ops.py`.

- **Pathways.py**: Defines three types of pathways: Normal Pathway, Subsampled Pathway, and Fully Connected Pathway (consisting only of fully connected layers). The core function `makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM` creates layer objects as defined in `Layers.py`, connects the layers in a pathway, and sets inputs and outputs. The function `makeResidualConnectionBetweenLayersAndReturnOutput` (defined outside the class) creates residual connections between specific layers. For Subsampled Pathway, the function `upsampleOutputToNormalRes` upsamples the final output to match the shape of the normal pathway's output.

- **Cnn3d.py**: The `make_cnn_model` function creates objects of classes defined in `pathways.py`, constructs each pathway, and combines the outputs of normal and subsampled pathways to create the FcPathway. The final output of the model's forward propagation is obtained by creating the Softmax Layer.

- **Trainer.py**: Calls functions from `cost_functions.py` to define the cost function, and from `optimizers.py` to create the optimizer. It applies the optimizer and loss function to perform update operations, returning these operations to `cnn3d.py`.

- **Training.py**: The function `doTrainOrValidationOnBatchesAndReturnMeanAccuraciesOfSubepoch` executes the TensorFlow session and prints accuracy, dice score, etc. The `do_training` function controls the entire training process, loading images from memory for each subepoch and sampling them to create segments for TensorFlow session.

### Sampling and Model Input
- **Sampling.py**: Defines functions related to loading images and extracting segments. It involves random selection of images, sampling based on specified methods (foreground-background ratio), and intensity augmentation as set in the config files. The file also discusses the use of weightMaps for advanced sampling methods.

### ConfigParsing
This folder contains files responsible for reading configuration parameters from config files, organizing these parameters, and passing them to the `make_3d_cnn` function in `cnn3d.py` and the `do_training` function in `training.py`.

