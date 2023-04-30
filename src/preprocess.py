import torch
import os
import pandas as pd
import math
import torch.nn.functional as F
from PIL import Image

from dataset import PreloadedImagesDataset


def preprocess(data_dir_path, metadata_csv_path):
    dataset_image_path = f"{data_dir_path}.pt"
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
    torch.save(dataset, dataset_image_path)      
    return dataset


def main():
    preprocess("training_set", "HAM10000_metadata.csv")
    preprocess("testing_set", "HAM10000_test_metadata.csv")



if __name__ == '__main__':
    main()
