import cv2
from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Union


# just a function to load images
def img_loader(path: str):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


class BarcodeJPGDataset(Dataset):
    def __init__(self, 
                 job_id: str,
                 data_dir: Union[str, Path],
                 phase: str,
                 bar_processing: str,
                 df_f: Union[str, Path],
                 mode: str,
                 cv: bool,
                 cv_fold: int,
                 input_size: int = 224,
                 transform_images: bool = True,
                 dry: bool = False,
                 val_idx: int = None):
        self.job_id = job_id
        self.data_dir = data_dir
        self.phase = phase
        self.mode = mode

        self.loader = img_loader
        self.image_transform = self.image_transforms(input_size, phase, transform_images)

        self.class_map = {}
        self.samples = self.collect_samples(bar_processing, df_f, val_idx, cv, cv_fold, dry)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        barcode_tensor, image_path, class_id, sample_idx = self.samples[idx]

        # load image only if training is either multimodal or image-based
        if self.mode != 'bar':
            img = self.loader(image_path)
            image_tensor = self.image_transform(img)
        else:
            image_tensor = 0

        return barcode_tensor, image_tensor, class_id, sample_idx

    # set up transformations for images for training and validation
    def image_transforms(self, input_size: int, phase: str, transform_images: bool):
        if not transform_images:
            return transforms.Compose([])

        if phase == 'train':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    # get phase (i.e., train/val) samples
    def records(self, df_f: Union[Path, str], val_idx: int, cv: bool, cv_fold: int) -> dict:
        dfs = {}
        records = pd.read_csv(df_f, header=0, sep='\t')

        # split dataset into train/val; cv relies on indices assigned to samples beforehand
        if cv:
            if cv_fold == 0:  # loocv
                determ = records['val_idx'] != val_idx
            else:
                determ = records['fold'] != val_idx
        else:
            determ = records['dataset'] == 'train'
        
        dfs['train'] = records.loc[determ, :].copy()
        dfs['val'] = records.loc[~determ, :].copy()
        
        return dfs

    # get samples from dataset dataframe + set up class map
    def collect_samples(self, bar_processing: str, df_f: Union[Path, str],
                        val_idx: int, cv: bool, cv_fold: int, dry=False) -> list:
        dfs = self.records(df_f, val_idx, cv, cv_fold)
        samples = []

        # we rely on one hot encoded barcodes for both encodings
        # we then use an additional layer to transform them into sequential encoding
        barcodes_dir = os.path.join(self.data_dir, 'barcodes', bar_processing, 'one_hot_bar')
        images_dir = os.path.join(self.data_dir, 'images')

        records = dfs[self.phase]
        for class_idx, class_name in enumerate(sorted(dfs['train']['species_name'].unique())):
            # or do I? Why not refer to the train class map and ignore the val class map? YES I DO BC the val sample
            # needs to be assigned to correct class idx!
            species_name = class_name.replace(' ', '_')
            self.class_map[species_name] = class_idx
            for idx, rec in records[records['species_name'] == class_name].iterrows():
                sample_id = rec['record_id']
                phase = rec['dataset']

                bar_path = f"{barcodes_dir}/{phase}/{species_name}/{sample_id}.npy"
                img_path = glob(f"{images_dir}/{phase}/{species_name}/{sample_id}.*")[0]

                samples.append(self.collect_sample(bar_path, img_path,
                                                   class_idx, sample_id))

                if dry:
                    return samples

        return samples

    # load barcode if training should not only rely on images
    def collect_sample(self, bar_path, img_path, class_idx, sample_id):
        if self.mode != 'img':
            barcode = np.load(bar_path)
            barcode = torch.from_numpy(np.expand_dims(barcode, axis=1)).float()
            barcode = torch.permute(barcode, (2, 0, 1))
        else:
            barcode = 0

        return barcode, img_path, class_idx, sample_id
