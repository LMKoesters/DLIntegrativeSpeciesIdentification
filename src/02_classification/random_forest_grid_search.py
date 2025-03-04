import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import sys
import torch
from torchvision import models
from torch.utils.data import DataLoader
from typing import Union

from model_bar_resnet import BarcodeJPGBar
from model_bar_resnet_sequential_embedding import SequentialEmbeddingBarcodeJPGBar
from barcodejpg_dataset import BarcodeJPGDataset

from train import BarcodeJPGTrain


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", "-j", help="set job id for folder name (required)", required=True)
    parser.add_argument("--num-workers", "-w", help="workers for data loading (default: 4)", type=int, default=4)
    parser.add_argument("--root-dir", '-d', nargs='?', default='../..',
                        help='Folder in which data folder is located (default: granparent directory)')
    parser.add_argument("--processings", '-p', nargs='*', default=['aligned_barcode'])
    parser.add_argument("--encodings", '-e', nargs='*', default=['one_hot_bar'])
    parser.add_argument("--pretrainings", '-g', nargs='*', default=['sep'])
    return parser.parse_args(args=None if sys.argv[1:] else ['--help'])


class RFGridSearch:
    def __init__(self, job_id: str, results_dir: Union[Path, str], root_dir: Union[Path, str],
                 num_workers: int, bar_enc: str, bar_prep: str, pretrainings: list[str]):
        self.job_id = job_id
        self.num_workers = num_workers
        self.bar_encoding = bar_enc
        self.bar_processing = bar_prep
        self.pretrainings = pretrainings

        self.results_dir = results_dir
        self.root_dir = root_dir
        self.data_dir = f'{root_dir}/data/{job_id}'
        self.marker = self.marker_from_ds_info()

    # read marker for dataset from dataset_info.tsv in data directory
    def marker_from_ds_info(self) -> str:
        ds_info = pd.read_csv(f'{self.root_dir}/data/dataset_info.tsv', header=0, sep='\t')
        return ds_info[ds_info['job_id'] == self.job_id]['marker'].values[0]

    # create train dataloader
    def gen_dataloader(self, batch_size: int = 32):
        df_f = f'{self.data_dir}/records/merged/{self.job_id}_{self.marker}.tsv'
        barcode_jpg_dataset = BarcodeJPGDataset(self.job_id, f'{self.data_dir}/server', 'train',
                                                self.bar_processing, df_f, 'fused',
                                                False, 0, transform_images=False)
        dataloader = DataLoader(barcode_jpg_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        return dataloader

    # main method
    def run(self):
        out_dir = f"{self.results_dir}/{bar_prep}/{bar_enc}/tr1_sep"
        barjpg_train = BarcodeJPGTrain(self.job_id, self.results_dir, self.data_dir, 0,
                                       self.bar_encoding, self.bar_processing, ['rf'], self.pretrainings,
                                       self.marker, pd.DataFrame(columns=['barcode_processing', 'barcode_encoding']))
        device = barjpg_train.device
        dataloader = self.gen_dataloader()
        bar_model, img_model_inst = barjpg_train.setup_base_models('rf')
        bar_model = barjpg_train.load_done_model('round 2', 'sep:rf', 'bar', out_dir, bar_model)
        img_model = barjpg_train.load_done_model('round 2', 'sep:rf', 'img', out_dir, img_model_inst.model)
        self.train_rf(bar_model, img_model, dataloader, device)

    def train_rf(self, barcode_model: Union[BarcodeJPGBar, SequentialEmbeddingBarcodeJPGBar],
                 image_model: models.resnet50, dataloader, device, batch_size: int = 32):
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=100, stop=3000)]
        # Number of features to consider at every split
        max_features = ['sqrt', 'log2']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110)]
        max_depth += [None]
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10, 15]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf = RandomForestClassifier()
        # Random search of parameters, using 4-fold cross validation,
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=2, verbose=2,
                                       random_state=42, n_jobs=-1)

        outputs_fused_np = np.empty(shape=[len(dataloader.dataset), 4096])
        labels_np = np.empty(shape=[len(dataloader.dataset)])
        sample_idx_total = []

        for i, batch_data in enumerate(dataloader):
            barcodes, images, labels, sample_idxs = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
            sample_idx_total.extend(sample_idxs)

            barcodes = barcodes.to(device)
            images = images.to(device)

            with torch.no_grad():
                barcode_model.eval()
                image_model.eval()

                outputs_bar = barcode_model(barcodes)
                outputs_img = image_model(images)

                outputs_fused_np[i * batch_size:(i + 1) * batch_size] = torch.cat(
                    (outputs_bar, outputs_img), dim=1).cpu().numpy()
                labels_np[i * batch_size:(i + 1) * batch_size] = labels.data

        rf_random.fit(outputs_fused_np, labels_np)

        pd.DataFrame.from_dict(rf_random.best_params_).to_csv(f'{self.results_dir}/rf_grid_search_results_{self.job_id}.tsv')


args = get_args()

for bar_enc in args.encodings:
    for bar_prep in args.processings:
        RFGridSearch(args.job_id, f'../../results/{args.job_id}/traditional',
                     args.root_dir, args.num_workers,
                     bar_enc, bar_prep, args.pretrainings).run()
