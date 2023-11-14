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

