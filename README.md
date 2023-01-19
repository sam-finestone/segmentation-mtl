# Multi-Task Learning For The Segmentation Of Animal Images

Pytorch implementation of a multi-task learning model for detection and segmentation of different animal breeds.

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
3. Training the multi-task U-Net model



