import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from dataloader_mtl import MTLDataset
import models
import utils

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    transforms = transforms.Compose([transforms.ToTensor()])
        
    #Training Set
    batch_size = 32
    trainset = MTLDataset(type='train', transform=transforms)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    #Test Set
    testset = MTLDataset(type='test', transform=transforms)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    #Validation Set
    valset = MTLDataset(type='val', transform=transforms)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)

    #MTL Network
    net = models.MTLNet(c_in=3, c_out_seg=1)
    net = net.float()

    #Loss and optimiser
    bin_criterion = torch.nn.CrossEntropyLoss()
    seg_criterion = utils.DiceLoss()
    bbox_criterion = torch.nn.L1Loss()
    uncertainty_loss = utils.UncertaintyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.0001,betas=(0.9, 0.999), 
        eps=1e-08, weight_decay=0, amsgrad=False)

    print('Beginning training...')
    #Training epochs
    for epoch in range(1):
        net.train()
        bin_running_loss = 0.0
        seg_running_loss = 0.0
        bbox_running_loss = 0.0
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            images, bbox_labels, bin_labels, mask_labels = data

            optimizer.zero_grad()
            bin_pred, mask_pred, bbox_pred = net(images.float())

            mask_labels = mask_labels.squeeze(1)
            mask_pred = mask_pred.squeeze(1)

            #Compute losses
            bin_loss = bin_criterion(bin_pred, bin_labels)
            seg_loss = seg_criterion(mask_pred, mask_labels)
            bbox_loss = bbox_criterion(bbox_pred, bbox_labels)
            bbox_loss = bbox_loss.sum()

            #Uncertainty weighted loss
            total_loss = uncertainty_loss(seg_loss, bin_loss, bbox_loss)

            total_loss.backward()
            optimizer.step()

            bin_running_loss += bin_loss.item()
            seg_running_loss += seg_loss.item()
            bbox_running_loss += bbox_loss.item()
            running_loss += total_loss.item()

            #Print loss values every 20 mini-batches
            if (i + 1) % 20 == 0 : 
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                print(f'Classification Loss: {bin_running_loss/20}')
                print(f'Segmentation Loss: {seg_running_loss/20}')
                print(f'Bounding Box Loss: {bbox_running_loss/20}')
                bin_running_loss = 0.0
                seg_running_loss = 0.0
                bbox_running_loss = 0.0
                running_loss = 0.0
        
        #Calculating metrics on validation set
        net.eval()
        val_size = len(val_loader.dataset)
        bin_correct = 0.0
        dice_metric = 0.0
        iou_bbox = 0.0

        with torch.no_grad():
            for val_data in val_loader:
                images, bbox_labels, bin_labels, mask_labels = val_data
                bin_pred, mask_pred, bbox_pred = net(images.float())

                mask_labels = mask_labels.squeeze(1)
                mask_pred = mask_pred.squeeze(1)

                probs = torch.sigmoid(mask_pred)
                predicted_masks = (probs >= 0.5).float() * 1

                bin_correct += (bin_pred.argmax(1) == bin_labels).sum().item()
                dice_metric += utils.dice_score(predicted_masks, mask_labels)
                iou_bbox += utils.iou_metric(bbox_pred, bbox_labels)

        accuracy_bin = bin_correct / val_size
        dice_seg = dice_metric / val_size * batch_size
        final_iou_bbox = iou_bbox / val_size * batch_size

        print('Validation Metrics:\n---------------------------------------------')
        print(f'Dice Score - Segmentation: {round(dice_seg.item(),4)}')
        print(f'Accuracy - Classification: {round(accuracy_bin,4)}')
        print(f'IoU - Bounding Box Regression: {round(final_iou_bbox.item(),4)}')
        print('---------------------------------------------')

    print('Training done.')

    #Calculating metrics on test set
    net.eval()
    test_size = len(testloader.dataset)
    bin_correct = 0.0
    dice_metric = 0.0
    iou_bbox = 0.0

    with torch.no_grad():
        for test_data in testloader:
            images, bbox_labels, bin_labels, mask_labels = test_data
            bin_pred, mask_pred, bbox_pred = net(images.float())

            mask_labels = mask_labels.squeeze(1)
            mask_pred = mask_pred.squeeze(1)

            probs = torch.sigmoid(mask_pred)
            predicted_masks = (probs >= 0.5).float() * 1

            bin_correct += (bin_pred.argmax(1) == bin_labels).sum().item()
            dice_metric += utils.dice_score(predicted_masks, mask_labels)
            iou_bbox += utils.iou_metric(bbox_pred, bbox_labels)

    accuracy_bin = bin_correct / test_size
    dice_seg = dice_metric / test_size * batch_size
    final_iou_bbox = iou_bbox / test_size * batch_size

    print('Test Set Metrics:\n---------------------------------------------')
    print(f'Dice Score - Segmentation: {round(dice_seg.item(),4)}')
    print(f'Accuracy - Classification: {round(accuracy_bin,4)}')
    print(f'IoU - Bounding Box Regression: {round(final_iou_bbox.item(),4)}')
    print('---------------------------------------------')