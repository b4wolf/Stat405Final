import torch
import torch.nn as nn
import os
import pandas as pd
import random
import argparse
import torch.optim as optim
import numpy as np


from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
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


# Define a custom dataset to handle images and metadata
class CustomDataset(Dataset):
    def __init__(self, image_paths, metadata, preprocess=None):
        self.image_paths = image_paths
        self.metadata = metadata
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.preprocess:
            image = self.preprocess(image)
        return image, self.metadata[idx]


class MyDataset(Dataset):
    def __init__(self, combined_features_tensor, labels_list):
        self.data = combined_features_tensor
        self.label = labels_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]

        return data, label


def train(args, model, device, train_loader, valid_loader, criterion,optimizer, epoch):
    model.train()
    running_loss = 0.0
    weight_dtype = model.weight.dtype
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.to(device, dtype=weight_dtype)
        output = model(data)
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

    # Evaluate the model on the validation set
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.to(device, dtype=weight_dtype)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()

        valid_loss = valid_loss / len(valid_loader)
        print(f"Validation Loss: {valid_loss:.4f}")


def test(args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    weight_dtype = model.weight.dtype
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=weight_dtype), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_old(args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    y_true_list = []
    y_pred_list = []
    y_pred_prob_list = []
    weight_dtype = model.weight.dtype
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.to(device, dtype=weight_dtype)
            output = model(data)
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


def fill_dataset_nan_with_mean(dataset):
    # Find the dimensions of the dataset
    num_samples = len(dataset)
    num_features = dataset[0][0].shape[0]

    # Collect all feature values and create a mask for NaN values
    all_features = torch.empty((num_samples, num_features))
    nan_mask = torch.empty((num_samples, num_features), dtype=torch.bool)

    for i, (data, _) in enumerate(dataset):
        all_features[i] = data
        nan_mask[i] = torch.isnan(data)

    # Calculate the mean of each feature without considering NaN values
    mean_vals = torch.nanmean(all_features, dim=0)

    # Fill the NaN values with the corresponding mean values for each feature
    for i in range(num_features):
        all_features[:, i].masked_fill_(nan_mask[:, i], mean_vals[i])

    # Update the dataset with the new feature values
    for i, (data, target) in enumerate(dataset):
        dataset.data[i] = all_features[i]

def preprocess(device, data_dir_path, metadata_csv_path):
    if os.path.exists(f"{data_dir_path}.pt"):
        mydataset = torch.load(f"{data_dir_path}.pt")
    else:
        # Load pre-trained CNN model
        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        model = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Remove the fully connected layer
        model = nn.Sequential(*list(model.children())[:-1])
        model = model.to(device)
        metadata_list = []
        image_paths = []

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
        for image_filename in os.listdir(data_dir_path):
            if image_filename.startswith("ISIC"):
                image_id = os.path.splitext(image_filename)[0]
                image_paths.append(os.path.join(data_dir_path, image_filename))
                metadata = metadata_dict.get(image_id)
                metadata_list.append(metadata[1:])
                label_list.append(metadata[0])
        

        # Create custom dataset and dataloader
        dataset = CustomDataset(image_paths, metadata_list, preprocess)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

        # Extract feature vectors and concatenate metadata
        combined_features_list = []
        for images, metadata in dataloader:
            images = images.to(device)
            with torch.no_grad():
                image_features = model(images)
                image_features = image_features.view(image_features.size(0), -1) # Flatten features

            metadata_tensor = torch.stack(metadata, dim=1).to(device)
            combined_features = torch.cat([image_features, metadata_tensor], dim=1)
            combined_features_list.append(combined_features)

            
        combined_features_tensor = torch.cat(combined_features_list, dim=0)
        mydataset = MyDataset(combined_features_tensor, label_list)

        torch.save(mydataset, f'{data_dir_path}.pt')
    
    fill_dataset_nan_with_mean(mydataset)
    print("Loaded data")
    # print(mydataset.data)
    return mydataset




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
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    if (not torch.cuda.is_available()):
         raise OSError("Torch cannot find a cuda device")
    
    torch.manual_seed(args.seed)

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = preprocess(device, "training_set", "HAM10000_metadata.csv")
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_set, valid_set = random_split(dataset, [train_size, valid_size])

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_set, batch_size= args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size= args.batch_size, shuffle=False, num_workers=0)

    test_dataset = preprocess(device, "test_data", "HAM10000_test_metadata.csv")
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=0)

    output_size = 7 # Number of output classes
    input_size = 2053
    fc_layer = nn.Linear(input_size, output_size).to(device)
    model = fc_layer.to(device)
    # model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, valid_loader, criterion, optimizer, epoch)
        test(args, model, device, criterion, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "ham.pt")

if __name__ == '__main__':
    main()
