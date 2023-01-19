# Multi-Task Learning For The Segmentation Of Animal Images

Pytorch implementation of a multi-task learning model for detection and segmentation of different animal breeds.
All files are currently set up to run on CPU only (will need to be adapted to run on GPU)

In this paper, the data is prepared and preprocessed for the task of image segmentation with image classification and object detection as auxiliary tasks. A simultaneous learning approach with hard parameter sharing is chosen as it is known to perform well when tasks are closely related, and it reduces the risk of overfitting and is less time consuming to implement. A data loading pipeline is implemented and the data is re-chunked so that images can be retrieved efficiently without loading all of the data into memory. The model architecture is based on U-Net, which is commonly used in image segmentation applications, including for MTL. During model training, the loss functions are combined using an equal weighted sum to form the overall model loss. The performance of the models is evaluated on the held-out test set using the Dice coefficient.
## Data preparation 

Used the Oxford-IIIT Pet dataset. The dataset consists of 37 categories of pets with around 200 photos for each breed. The scale, pose, and lighting of the animal images vary greatly. Each image has a ground truth annotation that describes the breed, the head ROI, and the pixel level trimap segmentation.

Training, validation and test data split: 80-10-10

## Data Preprocessing

Started by implementing a data loading pipeline. To implement an efficient dataloader for the algorithm, the train, test and validation sets
were re-chunked so that images can be retrieved efficiently without loading all of the data into memory. 


## Setup & Training 

1. Install all the dependencies:

```
pip install -r /requirements.txt
```

2. Clone this repo:

```
git clone https://github.com/sam-finestone/segmentation-mtl.git
```

3. Setup & Training
'''
- data_rechunk.py:
	Needs to be run first to re-chunk the data for efficient dataloading.
	Requires datasets, e.g. train images, to be stored as follows in working directory:
	'datasets/data_new/train/images.h5'

-dataloader_mtl.py: 
	Needs data_rechunk.py to be run before it can be run

-train_ ...py
	Training scripts for all models. Prints out model results and metrics.
'''  
  
4.Visualizations

'''
-mtl_vis.py
	File that visualises predicted mask vs target mask (mask.png) (requires matplotlib and mtl_model.pt)

-mtl_model.pt
	Fully trained MTL network that is requireds to run mtl_vis.py
'''

## Helper files 

'''
-utils.py:
	Utils script containing various functions and class for training and evaluation

-oeq_...py
	Open ended question scripts

-models.py
	Script containing all models necessary for training scripts
'''

