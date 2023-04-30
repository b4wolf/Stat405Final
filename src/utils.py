import numpy as np
import torch

from sklearn.metrics import roc_auc_score, recall_score

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

def class_auc(y_true, y_pred_prob, num_classes):
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
