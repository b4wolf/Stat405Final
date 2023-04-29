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

from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision import models
from torchvision.models import resnet50, resnet152, ResNet50_Weights, ResNet152_Weights, swin_v2_b, Swin_V2_B_Weights, convnext_base, ConvNeXt_Base_Weights
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image
from sklearn.metrics import roc_auc_score

# Functions to calculate accuracy and balanced accuracy
def accuracy(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    correct = torch.sum(y_true == y_pred).item()
    total = y_true.shape[0]
    return correct / total

def balanced_accuracy(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    num_classes = torch.unique(y_true).size(0)
    per_class_accuracies = []

    for i in range(num_classes):
        true_class = y_true == i
        pred_class = y_pred == i
        correct = torch.sum(true_class & pred_class).item()
        per_class_accuracies.append(correct / torch.sum(true_class).item())
    print(per_class_accuracies)

    return sum(per_class_accuracies) / num_classes

def per_category_auc(y_true, y_pred_prob, num_classes):
    y_true_np = y_true
    y_true_one_hot = np.eye(num_classes)[y_true_np]
    auc_scores = []

    for i in range(num_classes):
        try:
            auc = roc_auc_score(y_true_one_hot[:, i], y_pred_prob[:, i])
            auc_scores.append(auc)
        except ValueError:
            pass

    return auc_scores

class ModifiedNN(nn.Module):
    def __init__(self):
        super(ModifiedNN, self).__init__()

    def forward(self, x, metadata):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, metadata), dim=1)
        x = self.fc(x)
        return x
    
class ModifiedResNet(ModifiedNN):
    def __init__(self, resnet, transform, metadata_size):
        super(ModifiedResNet, self).__init__()
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features + metadata_size, 7)
        self.transform = transform


class ModifiedSwinTransformer(ModifiedNN):
    def __init__(self, swin_transformer, transform, metadata_size):
        super(ModifiedSwinTransformer, self).__init__()
        self.model = nn.Sequential(*list(swin_transformer.children())[:-1])
        self.fc = nn.Linear(swin_transformer.head.in_features + metadata_size, 7)
        self.transform = transform

class ModifiedConvNext(ModifiedNN):
    def __init__(self, convnext, transform, metadata_size):
        super(ModifiedConvNext, self).__init__()
        self.model = nn.Sequential(*list(convnext.children())[:-1])
        self.fc = nn.Linear(convnext.classifier[-1].in_features + metadata_size, 7)
        self.transform = transform

# Define a custom dataset to handle images and metadata
class PreloadedImagesDataset(Dataset):
    def __init__(self, images, metadata, labels):
        self.images = images
        self.metadata = metadata
        self.labels = labels
        self.data_augmentation = None
    
    def fill_missing_values_with_static(self, fill_value):
        metadata_arr = np.array(self.metadata, dtype=np.float32)
        metadata_arr[np.isnan(metadata_arr)] = fill_value
        self.metadata = metadata_arr
    
    def normalize_age(self):
        age = self.metadata[:, 5]
        self.metadata[:, 5] = age / 100
    
    def fill_missing_values_with_mean(self):
        metadata_arr = np.array(self.metadata, dtype=np.float32)
        self.metadata = metadata_arr
        age = self.metadata[:, 5]
        age_mean = np.nanmean(age)
        age[np.isnan(age)] = age_mean
        self.metadata[:, 5] = age

    def preprocess(self, preprocess):
        for idx, img in enumerate(self.images):
            self.images[idx] = preprocess(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.data_augmentation is not None:
            image = self.data_augmentation(self.images[idx])
        else:
            image = self.images[idx]
        return image, self.metadata[idx], self.labels[idx]



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
    correct = 0

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

            correct += pred.eq(target.view_as(pred)).sum().item()

            y_true_list.extend(target.cpu().numpy())
            y_pred_list.extend(pred.squeeze().cpu().numpy())
            y_pred_prob_list.extend(pred_prob.cpu().numpy())

    test_loss /= len(test_loader)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_pred_prob = np.array(y_pred_prob_list)

    # acc = accuracy(y_true, y_pred) * 100
    balanced_acc = balanced_accuracy(y_true, y_pred)
    category_auc = per_category_auc(y_true, y_pred_prob, num_classes=len(torch.unique(torch.tensor(y_true))))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Balanced Accuracy: {:.4f}, AUC for each category: {}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        balanced_acc,
        category_auc
    ))

    return test_loss, balanced_acc


def preprocess(data_dir_path, metadata_csv_path, model_name=''):
    saved_image_data_name = f"{data_dir_path}.pt"
    if os.path.exists(saved_image_data_name):
        dataset = torch.load(saved_image_data_name)
    else:   
        metadata_df = pd.read_csv(metadata_csv_path)

        metadata_df = metadata_df[['image_id', 'dx', 'dx_type', 'age', 'sex', 'localization', 'dataset']]
        metadata_dict = {row['image_id']: row[1:].tolist() for _, row in metadata_df.iterrows()}
        metadata_transposed = list(zip(*metadata_df.values.tolist()))
        mapping_list = [None, {'bcc': 0, 'vasc': 1, 'df': 2, 'bkl': 3, 'akiec': 4, 'mel': 5, 'nv': 6}, 
                        {'confocal': 0, 'follow_up': 1, 'consensus': 2, 'histo': 3}, 
                        {}, 
                        {'male': 0, 'female': 1, 'unknown': 2}, 
                        {'scalp': 0, 'face': 1, 'ear': 2, 'acral': 3, 'back': 4, 'foot': 5, 'lower extremity': 6, 'genital': 7, 'neck': 8, 'upper extremity': 9, 'hand': 10, 'chest': 11, 'unknown': 12, 'trunk': 13, 'abdomen': 14}, 
                        {'vienna_dias': 0, 'rosendahl': 1, 'vidir_molemax': 2, 'vidir_modern': 3, 'external': 4}]
        # Map the metadata using the corresponding dictionary for each column (except the first one) and list comprehensions
        mapped_metadata_transposed = [metadata_transposed[0]]  # Keep the first column (image_id) unchanged
        for column, mapping in zip(metadata_transposed[1:], mapping_list[1:]):
            mapped_column = [mapping.get(x, x) for x in column]
            mapped_metadata_transposed.append(mapped_column)

        # Transpose the mapped metadata back to the original shape
        metadata_int = list(zip(*mapped_metadata_transposed))
        metadata_dict = {row[0]: row[1:] for row in metadata_int}
        label_list = []
        image_list = []
        metadata_list = []
        for idx, image_filename in enumerate(os.listdir(data_dir_path)):
            if image_filename.startswith("ISIC"):
                image_id = os.path.splitext(image_filename)[0]
                image_path = os.path.join(data_dir_path, image_filename)
                image = Image.open(image_path).convert('RGB')
                image_list.append(image)
                metadata = metadata_dict.get(image_id)

                metadata_one_hot = []

                for meta_idx, x in enumerate(metadata[1:]):
                    if meta_idx != 1:
                        class_num = len(mapping_list[2+meta_idx])
                        if math.isnan(x):
                            x = class_num
                        one_hot_expression = F.one_hot(torch.tensor([x]), num_classes=class_num + 1).tolist()
                        metadata_one_hot.extend(one_hot_expression[0])
                    else:
                        metadata_one_hot.append(x)

                metadata_list.append(metadata_one_hot)
                label_list.append(metadata[0])
                if idx % 1000 == 0:
                    print(f"loaded {idx} images")

        # Create custom dataset and dataloader
        dataset = PreloadedImagesDataset(image_list, metadata_list, label_list)
        torch.save(dataset, saved_image_data_name)      

    
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
    
    parser.add_argument('--mode', type=str, default='all', metavar='MO', help='what is the mode to run this script? <all|preprocess>')
    
    parser.add_argument('--dataset', type=str, default='')

    parser.add_argument('--model', type=str, default='resnet-152', metavar='MD',
                        help='Which model to train')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    
    args = parser.parse_args()
    if args.mode == 'preprocess':
        if args.dataset.startswith('training_set'):
            preprocess(args.dataset, "HAM10000_metadata.csv", model_name='')
        elif args.dataset.startswith('testing_set'):
            preprocess(args.dataset, "HAM10000_test_metadata.csv", model_name='')

        return
 
    if (not torch.cuda.is_available()):
        raise OSError("Torch cannot find a cuda device")
    
    torch.manual_seed(args.seed)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    model_name = args.model
    train_set = preprocess("training_set", "HAM10000_metadata.csv", model_name)
    test_set = preprocess("testing_set", "HAM10000_test_metadata.csv", model_name)


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
    overfit_counter = 0
    patience = 60
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, criterion, optimizer, epoch)
        test_loss, mean_recall = test(args, model, device, criterion, test_loader)
        scheduler.step()
        if abs(train_loss - last_train_loss) < 0.001:
            convg_counter += 1
        else:
            convg_counter = 0
        last_train_loss = train_loss
        if mean_recall > best_mean_recall:
            best_mean_recall = mean_recall
            torch.save(model.state_dict(), saved_model_name)
            overfit_counter = 0
        else:
            overfit_counter += 1
        
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        mean_recall_history.append(mean_recall)
        if convg_counter > patience or overfit_counter > patience:
            print(f"Early stopping after {epoch} epochs, convg: {convg_counter}, overfit: {overfit_counter}")
            break
    print(train_loss_history)
    print(test_loss_history)
    print(mean_recall_history)

if __name__ == '__main__':
    main()
