# Multi-Task Learning For The Segmentation Of Animal Images

Pytorch implementation of a multi-task learning model for detection and segmentation of different animal breeds.

## Data preparation 

Used the Oxford-IIIT Pet dataset.

Training, validation and test data split: 80-10-10

## Data Preprocessing

Started by implementing a data loading pipeline. To implement an efficient dataloader for the algorithm, the train, test and validation sets
were re-chunked so that images can be retrieved efficiently without loading all of the data into memory. 




