import h5py
import numpy as np

#Re-chunking files so 1 label/image = 1 chunk

datasets = ['train','test','val']

for set in datasets:
    with h5py.File(f'datasets/data_new/{set}/images.h5', 'a') as images:
        image_chunks = images.create_dataset('chunked_images', images['images'].shape, chunks=(1,256,256,3), data=images['images'])

    with h5py.File(f'datasets/data_new/{set}/bboxes.h5', 'a') as bboxes:
        bbox_chunks = bboxes.create_dataset('chunked_bboxes', bboxes['bboxes'].shape, chunks=(1,4), data=bboxes['bboxes'])

    with h5py.File(f'datasets/data_new/{set}/binary.h5', 'a') as binary:
        binary_chunks = binary.create_dataset('chunked_binary', binary['binary'].shape, chunks=(1,1), data=binary['binary'])

    with h5py.File(f'datasets/data_new/{set}/masks.h5', 'a') as masks:
        mask_chunks = masks.create_dataset('chunked_masks', masks['masks'].shape, chunks=(1,256,256,1), data=masks['masks'])
        
        
    

    