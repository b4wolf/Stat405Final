import numpy as np
import torch
import sys
import os
import torch.nn as nn
import json
import hashlib
from torchvision import transforms
from torchvision import models
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader

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

# Functions to calculate accuracy and balanced accuracy
def accuracy(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    correct = torch.sum(y_true == y_pred).item()
    total = y_true.shape[0]
    return correct / total

def balanced_accuracy(y_true, y_pred):
    recall_per_class = recall_score(y_true, y_pred, average=None)

    # Calculate the mean recall (macro-averaged recall)
    mean_recall = np.mean(recall_per_class)

    return mean_recall, recall_per_class

def per_class_auc(y_true, y_pred_prob, num_classes):
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

def ensemble_prediction(models, img, metadata):
    pred = []
    for model in models:
        model.eval()
        output = model(img, metadata)
        pred_prob = torch.softmax(output, dim=1)  # get the predicted probabilities
        pred.append(pred_prob)
    # Calculate the average probability for each class
    pred_arr = torch.stack(pred)
    avg_probabilities = torch.mean(pred_arr, dim=0)
    pred = avg_probabilities.argmax(dim=1, keepdim=True)
    return pred, avg_probabilities




def eval(models, device, test_loader):
    y_true_list = []
    y_pred_list = []
    y_pred_prob_list = []
    weight_dtype = torch.float32
    with torch.no_grad():
        for data, metadata, target in test_loader:
            data, metadata, target = data.to(device, dtype=weight_dtype), metadata.to(device, dtype=weight_dtype), target.to(device)
            pred, pred_prob = ensemble_prediction(models, data, metadata)
            y_true_list.extend(target.cpu().numpy())
            y_pred_list.extend(pred.squeeze().cpu().numpy())
            y_pred_prob_list.extend(pred_prob.cpu().numpy())
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_pred_prob = np.array(y_pred_prob_list)

    acc = accuracy(y_true, y_pred) * 100
    balanced_acc, per_class_accuracy = balanced_accuracy(y_true, y_pred)
    num_classes = len(torch.unique(torch.tensor(y_true)))
    per_class_auc = per_class_auc(y_true, y_pred_prob, num_classes=num_classes)
    
    fpr = dict()
    tpr = dict()
    # Binarize the true labels
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])

    cm = confusion_matrix(y_true, y_pred)

    print('\nTest set:  Accuracy: ({:.0f}%), Balanced Accuracy: {:.4f}, Recall for each Class: {}, AUC for each category: {}, Confusion Matrix: {}\n'.format(
        acc,
        balanced_acc,
        per_class_accuracy,
        per_class_auc,
        cm
    ))

    result = {}
    result['accuracy'] = acc
    result['balanced_accuracy'] = balanced_acc
    result['per_class_accuracy'] = per_class_accuracy
    result['per_class_auc'] = per_class_auc
    result['confusion_matrix'] = cm
    result['fpr'] = fpr
    result['tpr'] = tpr
    return result




def main():
    modelList = sys.argv[1]
    test_set = torch.load('testing_set.pt')
    test_set_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.7635, 0.5461, 0.5705], std=[0.0896, 0.1183, 0.1330]),
        ])
    test_set.preprocess(test_set_preprocess)
    test_set.fill_missing_values_with_mean()
    test_set.normalize_age()
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    if (not torch.cuda.is_available()):
        raise OSError("Torch cannot find a cuda device")

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    model_list = [] 
    model_name_list = []
    with open(modelList, 'r') as f:
        for line in f:
            model_name = line.strip()
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
            else:
                exit(1)
            model = model.to(device)
            model_list.append(model)
            model_name_list.append(model_name)
    result_dict = eval(model_list, device, test_loader)
    result_dict['model_names'] = model_name_list

    # Convert the list of strings into a single string
    combined_string = ''.join(model_name_list)

    # Create a hash of the combined string using hashlib
    hash_object = hashlib.sha256(combined_string.encode('utf-8'))
    hash_hex = hash_object.hexdigest() % 100

    # Save the dictionary to a JSON file
    with open(f'result_{hash_hex}.json', 'w') as f:
        json.dump(result_dict, f)



    




if __name__ == '__main__':
    main()
