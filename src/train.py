import torch
import torch.nn as nn
import os
import pandas as pd
import random
import argparse
import torch.optim as optim
import numpy as np
import math
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision import models
from PIL import Image
from dataset import PreloadedImagesDataset
from modified_model import ModifiedResNet, ModifiedConvNext, ModifiedSwinTransformer
from utils import accuracy, balanced_accuracy, class_auc


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    weight_dtype = torch.float32
    for batch_idx, (data, metadata, target) in enumerate(train_loader):
        data, metadata, target = data.to(device, dtype=weight_dtype), metadata.to(device, dtype=weight_dtype), target.to(device)
        optimizer.zero_grad()
        output = model(data, metadata)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    # Calculate the average loss for this epoch
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch}/{args.epochs}, Loss: {epoch_loss:.4f}")


    return epoch_loss


def test(args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    # correct = 0

    y_true_list = []
    y_pred_list = []
    y_pred_prob_list = []
    weight_dtype = torch.float32
    with torch.no_grad():
        for data, metadata, target in test_loader:
            data, metadata, target = data.to(device, dtype=weight_dtype), metadata.to(device, dtype=weight_dtype), target.to(device)
            output = model(data, metadata)
            test_loss += criterion(output, target).item()  # sum up batch loss

            pred_prob = torch.softmax(output, dim=1)  # get the predicted probabilities
            pred = pred_prob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            y_true_list.extend(target.cpu().numpy())
            y_pred_list.extend(pred.squeeze().cpu().numpy())
            y_pred_prob_list.extend(pred_prob.cpu().numpy())

    test_loss /= len(test_loader)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_pred_prob = np.array(y_pred_prob_list)

    acc = accuracy(y_true, y_pred) * 100
    balanced_acc, per_class_acc = balanced_accuracy(y_true, y_pred)
    per_class_auc = class_auc(y_true, y_pred_prob, num_classes=len(torch.unique(torch.tensor(y_true))))

    print('\nTest set:  Test_loss: {:.0f}, Accuracy: ({:.0f}%), Balanced Accuracy: {:.4f}, Recall for each Class: {}, AUC for each category: {}\n'.format(
        test_loss,
        acc,
        balanced_acc,
        per_class_acc,
        per_class_auc,
    ))

    return test_loss, balanced_acc


def load_dataset(dataset_name):
    saved_dataset_image = f"{dataset_name}.pt"
    if os.path.exists(saved_dataset_image):
        dataset = torch.load(saved_dataset_image)
    else:   
        print("Dataset not exits")
        exit(1)      

    dataset.fill_missing_values_with_mean()
    dataset.normalize_age()
    return dataset


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')


    parser.add_argument('--model', type=str, default='resnet-152', metavar='MD',
                        help='Which model to train')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    
    args = parser.parse_args()
 
    if (not torch.cuda.is_available()):
        raise OSError("Torch cannot find a cuda device")
    
    torch.manual_seed(args.seed)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    model_name = args.model
    train_set = load_dataset("training_set")
    test_set = load_dataset("testing_set")


    data_augmentation = transforms.Compose([
        transforms.Resize((600, int(400 * 1.25))),
        transforms.RandomResizedCrop(
            size=224,
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0)
        ),
        transforms.RandomChoice([
            transforms.RandomRotation(0),
            transforms.RandomRotation(90),
            transforms.RandomRotation(180),
            transforms.RandomRotation(270)
        ]),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0896, 0.1183, 0.1330]),
    ])
    train_set.data_augmentation = data_augmentation
    test_set_preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0896, 0.1183, 0.1330]),
    ])
    test_set.preprocess(test_set_preprocess)
    
    num_samples_per_class = np.zeros(7)
    for label in train_set.labels:
        num_samples_per_class[label] += 1
    
    num_samples_per_class_inversed = 1 / num_samples_per_class
    weighted_arr = num_samples_per_class_inversed / np.sum(num_samples_per_class_inversed)
    weighted_tensor = torch.from_numpy(weighted_arr.astype('float32')).to(device)

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_set, batch_size= args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if model_name == 'resnet-50':
        weights = models.ResNet50_Weights.DEFAULT
        model = ModifiedResNet(models.resnet50(weights=weights), weights.transforms(), 32)
    elif model_name == 'resnet-152':
        weights = models.ResNet152_Weights.DEFAULT
        model = ModifiedResNet(models.resnet152(weights=weights), weights.transforms(), 32)
    elif model_name == 'swin-b':
        weights = models.Swin_V2_B_Weights.DEFAULT
        model = ModifiedSwinTransformer(models.swin_v2_b(weights=weights), weights.transforms(), 32)
    elif model_name == 'swin-s':
        weights = models.Swin_V2_S_Weights.DEFAULT
        model = ModifiedSwinTransformer(models.swin_v2_s(weights=weights), weights.transforms(), 32)
    elif model_name == 'swin-t':
        weights = models.Swin_V2_T_Weights.DEFAULT
        model = ModifiedSwinTransformer(models.swin_v2_t(weights=weights), weights.transforms(), 32)
    elif model_name == 'convnext-l':
        weights = models.ConvNeXt_Large_Weights.DEFAULT
        model = ModifiedConvNext(models.convnext_large(weights=weights),  weights.transforms(), 32)
    elif model_name == 'convnext-b':
        weights = models.ConvNeXt_Base_Weights.DEFAULT
        model = ModifiedConvNext(models.convnext_base(weights=weights),  weights.transforms(), 32)
    elif model_name == 'convnext-s':
        weights = models.ConvNeXt_Small_Weights.DEFAULT
        model = ModifiedConvNext(models.convnext_small(weights=weights),  weights.transforms(), 32)
    elif model_name == 'convnext-t':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = ModifiedConvNext(models.convnext_tiny(weights=weights), weights.transforms(), 32)
    else:
        exit(1)

    saved_model_name = f"ham_{model_name}.pt"
    if os.path.exists(saved_model_name):
        saved_model_state = torch.load(saved_model_name)
        model.load_state_dict(saved_model_state)
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weighted_tensor)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)


    test_loss, mean_recall = test(args, model, device, criterion, test_loader)
    last_train_loss = float('inf')
    best_mean_recall = mean_recall
    train_loss_history = []
    test_loss_history = []
    mean_recall_history = []
    convg_counter = 0
    patience = 60
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, criterion, optimizer, epoch)
        test_loss, mean_recall = test(args, model, device, criterion, test_loader)
        scheduler.step()
        if abs(train_loss - last_train_loss) < 0.01:
            convg_counter += 1
        else:
            convg_counter = 0
        last_train_loss = train_loss
        if mean_recall > best_mean_recall:
            best_mean_recall = mean_recall
            torch.save(model.state_dict(),  saved_model_name)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        mean_recall_history.append(mean_recall)
        if convg_counter > patience:
            print(f"Early stopping after {epoch} epochs, convg: {convg_counter}")
            break
    print(train_loss_history)
    print(test_loss_history)
    print(mean_recall_history)

if __name__ == '__main__':
    main()
