import numpy as np
from torch.utils.data import Dataset

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