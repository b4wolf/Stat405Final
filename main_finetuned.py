import torch
import torch.nn as nn
import os
import pandas as pd
import random
import argparse
import torch.optim as optim
import numpy as np
import time


from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.models import resnet50, resnet152, ResNet50_Weights, ResNet152_Weights, swin_v2_b, Swin_V2_B_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.metrics import roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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


class ModifiedResNet152(nn.Module):
    def __init__(self, resnet152, metadata_size):
        super(ModifiedResNet152, self).__init__()
        self.resnet152 = nn.Sequential(*list(resnet152.children())[:-1])
        self.fc = nn.Linear(resnet152.fc.in_features + metadata_size, 7)

    def forward(self, x, metadata):
        x = self.resnet152(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, metadata), dim=1)
        x = self.fc(x)
        return x

class ModifiedSwinTransformer(nn.Module):
    def __init__(self, swin_transformer, metadata_size):
        super(ModifiedSwinTransformer, self).__init__()
        self.swin_transformer = nn.Sequential(*list(swin_transformer.children())[:-1])
        self.fc = nn.Linear(swin_transformer.head.in_features + metadata_size, 7)

    def forward(self, x, metadata):
        x = self.swin_transformer(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, metadata), dim=1)
        x = self.fc(x)
        return x



# Define a custom dataset to handle images and metadata
class PreloadedImagesDataset(Dataset):
    def __init__(self, images, metadata, labels):
        self.images = images
        self.metadata = metadata
        self.labels = labels
    
    def fill_missing_values_with_static(self, fill_value):
        metadata_arr = np.array(self.metadata, dtype=np.float32)
        metadata_arr[np.isnan(metadata_arr)] = fill_value
        self.metadata = metadata_arr

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.metadata[idx], self.labels[idx]



def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    weight_dtype = model.resnet50[0].weight.dtype
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
    print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss:.4f}")

    # # Evaluate the model on the validation set
    # model.eval()
    # valid_loss = 0.0
    # with torch.no_grad():
    #     for inputs, metadata, labels in valid_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         inputs = inputs.to(device, dtype=weight_dtype)
    #         outputs = model(inputs, metadata)
    #         loss = criterion(outputs, labels)

    #         valid_loss += loss.item()

    #     valid_loss = valid_loss / len(valid_loader)
    #     print(f"Validation Loss: {valid_loss:.4f}")


def test_old(args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    weight_dtype = model.resnet50[0].weight.dtype
    with torch.no_grad():
        for data, metadata, target in test_loader:
            data, metadata, target = data.to(device, dtype=weight_dtype), metadata.to(device, dtype=weight_dtype), target.to(device)
            output = model(data, metadata)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test(args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    y_true_list = []
    y_pred_list = []
    y_pred_prob_list = []
    weight_dtype = model.resnet50[0].weight.dtype
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

    test_loss /= len(test_loader.dataset)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_pred_prob = np.array(y_pred_prob_list)

    acc = accuracy(y_true, y_pred)
    balanced_acc = balanced_accuracy(y_true, y_pred)
    category_auc = per_category_auc(y_true, y_pred_prob, num_classes=len(torch.unique(torch.tensor(y_true))))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Balanced Accuracy: {:.4f}, AUC for each category: {}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        balanced_acc,
        category_auc
    ))


def preprocess(data_dir_path, metadata_csv_path, model_name):
    saved_preprocessed_data_name = f"{data_dir_path}_{model_name}.pt"
    if os.path.exists(saved_preprocessed_data_name):
        dataset = torch.load(saved_preprocessed_data_name)
    else:   
        if model_name == 'resnet-152':
            preprocess = ResNet50_Weights.DEFAULT.transforms()
        elif model_name == 'swin':
            preprocess = Swin_V2_B_Weights.DEFAULT.transforms()
        # Read the metadata CSV file
        # metadata_csv_path = 'HAM10000_metadata.csv'
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
                image_list.append(preprocess(image))
                metadata = metadata_dict.get(image_id)
                metadata_list.append(metadata[1:])
                label_list.append(metadata[0])
                if idx % 1000 == 0:
                    print(f"loaded {idx} images")
    

        # Create custom dataset and dataloader
        dataset = PreloadedImagesDataset(image_list, metadata_list, label_list)
        torch.save(dataset, saved_preprocessed_data_name)      
    dataset.fill_missing_values_with_static(-1)
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
    train_set = preprocess("training_set", "HAM10000_metadata.csv", model_name)
    # Split the dataset into training and validation sets
    # train_size = int(0.8 * len(dataset))
    # valid_size = len(dataset) - train_size
    # train_set, valid_set = random_split(dataset, [train_size, valid_size])

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_set, batch_size= args.batch_size, shuffle=True, **kwargs)
    # valid_loader = DataLoader(valid_set, batch_size= args.batch_size, shuffle=False, **kwargs)

    test_set = preprocess("test_data", "HAM10000_test_metadata.csv", model_name)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # model = ModifiedResNet50(resnet50(weights=ResNet50_Weights.DEFAULT), 5).to(device)

    saved_model_name = f"ham_{model_name}.pt"
    if os.path.exists(saved_model_name):
        model = torch.load(saved_model_name)
    else:
        if model_name == 'resnet-152':
            model = ModifiedResNet152(resnet152(weights=ResNet152_Weights.DEFAULT), 5)
        elif model_name == 'swin':
            model = ModifiedSwinTransformer(swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT), 5)
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test(args, model, device, criterion, test_loader)
        scheduler.step()

        if epoch % 5 == 0 and (args.save_model):
            torch.save(model.state_dict(), saved_model_name)

if __name__ == '__main__':
    main()
