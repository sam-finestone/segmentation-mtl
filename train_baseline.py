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

    #Baseline Segmentation Network
    net = models.UNet(c_in=3, c_out_seg=1)
    net = net.float()

    #Loss and optimiser
    bin_criterion = torch.nn.CrossEntropyLoss()
    seg_criterion = utils.DiceLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.0001,betas=(0.9, 0.999), 
        eps=1e-08, weight_decay=0, amsgrad=False)

    print('Beginning training...')
    #Training epochs
    for epoch in range(1):
        net.train()
        seg_running_loss = 0.0
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            images, bbox_labels, bin_labels, mask_labels = data

            optimizer.zero_grad()
            mask_pred = net(images.float())

            mask_labels = mask_labels.squeeze(1)
            mask_pred = mask_pred.squeeze(1)

            #Compute losses
            seg_loss = seg_criterion(mask_pred, mask_labels)
            total_loss = seg_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            #Print loss values every 20 mini-batches
            if (i + 1) % 20 == 0 : 
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss/20))

                running_loss = 0.0
        
        #Calculating metrics on validation set
        net.eval()
        val_size = len(val_loader.dataset)
        dice_metric = 0.0

        with torch.no_grad():
            for val_data in val_loader:
                images, bbox_labels, bin_labels, mask_labels = val_data
                mask_pred = net(images.float())

                mask_labels = mask_labels.squeeze(1)
                mask_pred = mask_pred.squeeze(1)

                probs = torch.sigmoid(mask_pred)
                predicted_masks = (probs >= 0.5).float() * 1
                
                dice_metric += utils.dice_score(predicted_masks, mask_labels)

        dice_seg = dice_metric / val_size * batch_size

        print('Validation Metrics:\n---------------------------------------------')
        print(f'Dice Score - Segmentation: {round(dice_seg.item(),4)}')
        print('---------------------------------------------')

    print('Training done.')

    #Calculating metrics on test set
    net.eval()
    test_size = len(testloader.dataset)
    dice_metric = 0.0

    with torch.no_grad():
        for test_data in testloader:
            images, bbox_labels, bin_labels, mask_labels = test_data
            mask_pred = net(images.float())

            mask_labels = mask_labels.squeeze(1)
            mask_pred = mask_pred.squeeze(1)

            probs = torch.sigmoid(mask_pred)
            predicted_masks = (probs >= 0.5).float() * 1
            
            dice_metric += utils.dice_score(predicted_masks, mask_labels)

    dice_seg = dice_metric / test_size * batch_size

    print('Test Set Metrics:\n---------------------------------------------')
    print(f'Dice Score - Segmentation: {round(dice_seg.item(),4)}')
    print('---------------------------------------------')