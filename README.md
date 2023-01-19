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

```
- data_rechunk.py:
	Needs to be run first to re-chunk the data for efficient dataloading.
	Requires datasets, e.g. train images, to be stored as follows in working directory:
	'datasets/data_new/train/images.h5'

-dataloader_mtl.py: 
	Needs data_rechunk.py to be run before it can be run

-train_ ...py
	Training scripts for all models. Prints out model results and metrics.
```  
  
4.Visualizations

```
-mtl_vis.py
	File that visualises predicted mask vs target mask (mask.png) (requires matplotlib and mtl_model.pt)

-mtl_model.pt
	Fully trained MTL network that is requireds to run mtl_vis.py
```

## Helper files 

```
-utils.py:
	Utils script containing various functions and class for training and evaluation

-oeq_...py
	Open ended question scripts

-models.py
	Script containing all models necessary for training scripts
```

## Experiments

For the main part of the analysis, the evaluatation of the different MTL setups was done by comparing the image segmentation performance of the following models on the held out test set:

1. Full MTL Network: with target task of image segmentation and auxiliary tasks of binary classification and object detection
2. Baseline Segmentation Network: U-Net for image segmentation with no auxiliary tasks
3. Ablated MTL Network #1: with target task of image segmentation and auxiliary task of binary classification
4. Ablated MTL Network #2: with target task of image segmentation and auxiliary task of object detection

First, we examine the extent to which the auxiliary tasks of binary classification and object detection impact image segmentation performance. We compare the baseline U-Net architecture against the Full MTL network. For this experiment, we are interested in whether simultaneously learning binary classification and object detection boosts the segmentation performance, or whether negative transfer occurs.


## Results

Dice Score on held-out test set (after 60 epochs)
Model | Dice Score (%)
------------- | -------------
Full MTL model | 90.21
Baseline model | 92.94
Ablation model w/ classification | 88.04
Ablation model w/ bbox | 92.75
OEQ w/ uncertainty model | 87.73
OEQ w/ 80-10-10 model | 91.80
OEQ w/ 75-15-15 model | 91.54
OEQ w/ 50-25-25 model | 91.29
OEQ w/ 25-25-50 model | 90.32
OEQ w/ 25-50-25 model | 89.83


## Limitations

Limitations exist in this MTL study. As the dataset consists of images of cats and dogs, it cannot be certain that the trained model will generalise well to other animals. Furthermore, the results may not be replicable in different settings where image segmentation is of interest (e.g. in the medical domain). While the nature of MTL allows the use of a relatively small dataset and still produce an accurate network, improved performance may have been seen with a larger training set. Moreover, in the absence of computational resources for this project, it was unable to verify the statistical significance of the experimental results. In the future, this could be done by running each experiment multiple times and computing summary statistics.

## Future direction 

A future study on this particular MTL project would be to tackle a soft-parameter sharing approach. Assumptions were made about the relatedness of the tasks when choosing to use hard-parameter sharing, so it is possible that with an alternative MTL model different results would be seen. Furthermore, with more computational resources, more advanced network architectures could be attempted. With a larger set of auxiliary tasks, more could be learned about the data and what tasks are most beneficial to image segmentation.


## Conclusion 

It was set out to evaluate whether a multi-task learning setup would be beneficial and improve the target task of image segmentation of pet breed images. The baseline model was found to be the best performing model in regards to image segmentation. However, the ablation model with object detection came very close, with a 0.19% difference. The ablation model with binary classification performed less well, achieving 4.9% lower dice score than the baseline model. It is valuable to note that good performance was also seen on the auxiliary tasks. The full MTL model achieved a dice score of 90.21% on image segmentation, as well as high binary classification accuracy and relatively high performance with the object detection task. If further work were to be conducted, expanding the network to alternative tasks and a dataset with a wider variety of animals would be of interest. Additionally, a deeper investigation into the impact of loss function weighting would be desirable.



