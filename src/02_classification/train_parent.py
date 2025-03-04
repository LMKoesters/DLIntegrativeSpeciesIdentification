import sys

from datetime import datetime, timedelta
from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import time
from typing import Union

from train import BarcodeJPGTrain
from blast import Blast


# base class for some wrapper/meta functions for training; also used for traditional training directly
class TraditionalTraining:
    def __init__(self, results_dir: Union[Path, str], cross_validation: bool, args):
        self.job_id: str = args.job_id
        self.run_idx: int = args.run_index
        self.runs: int = args.runs
        self.batch_num: int = args.batch_num
        self.batches: int = args.batches
        self.num_epochs: int = args.num_epochs
        self.num_workers: int = args.num_workers
        self.hide_progress: bool = args.hide_progress
        self.root_dir: Union[Path, str] = args.root_dir
        self.classifiers: list[str] = args.classifiers
        self.bar_preps: list[str] = args.processings
        self.bar_encs: list[str] = args.encodings
        self.pretrainings: list[str] = args.pretrainings
        self.cv: bool = cross_validation
        self.cv_folds: int = args.cv_folds
        self.blast: bool = args.blast
        self.results_dir: Union[Path, str] = results_dir
        self.data_dir: str = f'{self.root_dir}/data/{self.job_id}'

        self.marker: str = self.marker_from_ds_info()
        self.results: pd.DataFrame = self.create_results(['barcode_processing', 'barcode_encoding',
                                                          'round_1', 'round_2', 'val_acc', 'val_loss',
                                                          'best_epoch', 'val_idx'])

    # read marker for dataset from dataset_info.tsv in data directory
    def marker_from_ds_info(self) -> str:
        ds_info = pd.read_csv(f'{self.root_dir}/data/dataset_info.tsv', header=0, sep='\t')
        return ds_info[ds_info['job_id'] == self.job_id]['marker'].values[0]

    # read results dataframe if it exists, otherwise create one
    def create_results(self, cols: list[str]) -> pd.DataFrame:
        result_file = f"{self.results_dir}/results_{self.run_idx}.tsv"
        if os.path.isfile(result_file):
            results = pd.read_csv(result_file, header=0, sep='\t')
        else:
            results = pd.DataFrame(columns=cols)
        return results

    # wrapper that iterates over barcode processings and encodings specified by user
    def run(self):
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
        for bar_enc in self.bar_encs:
            for bar_prep in self.bar_preps:
                self.train_models(bar_enc, bar_prep, self.pretrainings, self.classifiers)

        if self.blast:
            Blast(self.job_id, self.root_dir, self.marker, f'{self.data_dir}/records/merged/{self.job_id}_{self.marker}.tsv', self.num_workers).run_traditional()

    # initiates class for and starts training
    def train_models(self, bar_enc, bar_prep, pretrainings, classifiers, mode='fused', val_idx=None):
        barjpg_train = BarcodeJPGTrain(job_id=self.job_id, results_dir=self.results_dir, data_dir=self.data_dir,
                                       run_idx=self.run_idx, bar_encoding=bar_enc, bar_processing=bar_prep,
                                       classifiers=classifiers, pretrainings=pretrainings,
                                       marker=self.marker, results=self.results, cv_folds=self.cv_folds, cv=self.cv,
                                       mode=mode, num_workers=self.num_workers, num_epochs=self.num_epochs,
                                       val_idx=val_idx, hide_progress=self.hide_progress, root_dir=self.root_dir)
        barjpg_train.run()


# class for cross validation training that inherits from traditional training
# adds some functionality needed for cv
class CrossValidation(TraditionalTraining):
    def __init__(self, results_dir, cross_validation, args):
        super().__init__(results_dir, cross_validation, args)

        self.subset = args.subset

    # wrapper for training on different processings, images and fusion methods
    def run_preprocessing(self):
        # test barcode processings first - we can then choose best option for fusion
        for bar_enc in self.bar_encs:
            for bar_prep in self.bar_preps:
                self.train_samples(bar_enc, bar_prep, pretrainings=['sep'],
                                   classifiers=[], mode='bar')

        # barcode models are finished, now train on images separately
        self.train_samples('aligned_barcode', 'one_hot_bar', pretrainings=['sep'], classifiers=[], mode='img')
        
        if self.blast:
            Blast(self.job_id, self.root_dir, self.marker, f'{self.data_dir}/records/merged/{self.job_id}_{self.marker}_cv{self.cv_folds}.tsv', self.num_workers).run_cv()

        # reuse finished barcode + image models (i.e., feature extractors) for fusion
        # for barcodes: determine the best processing+encoding to reuse
        bar_enc, bar_prep = self.best_processing()
        print(f'Best processing methods: {bar_enc} & {bar_prep}')

        return bar_enc, bar_prep

    # train multimodal models
    def run_fusion(self, bar_encs, bar_preps):
        df = pd.concat([pd.read_csv(f, header=0, sep='\t') for f in glob(f'{self.root_dir}/results/{self.job_id}/traditional/results*.tsv')])
        df['mean_acc'] = df.groupby(['round_1', 'round_2'])['val_acc'].transform('mean')
        best_fused = df[df['mean_acc'] == max(df['mean_acc'])]
        round_1s = best_fused['round_1'].unique()
        round_2s = best_fused['round_2'].unique()
        print(f'Best fusion method(s): {round_1s} & {round_2s}')

        for bar_enc in bar_encs:
            for bar_prep in bar_preps:
                self.train_samples(bar_enc, bar_prep, round_1s, round_2s)

    # determine best processing/encoding of barcodes by consulting cv results
    def best_processing(self):
        df = pd.concat(
            [pd.read_csv(f, header=0, sep='\t') for f in glob(f'{self.results_dir}/results_*.tsv')])

        df = df.loc[df['round_2'] == 'bar', :].copy()
        df.drop_duplicates(subset=['barcode_processing', 'barcode_encoding', 'val_idx'],
                           inplace=True)
        df['mean_acc'] = df.groupby(['barcode_processing', 'barcode_encoding'])['val_acc'].transform('mean')
        df.drop_duplicates(subset=['barcode_processing', 'barcode_encoding'],
                           inplace=True)
        best_preps = df[df['mean_acc'] == max(df['mean_acc'])]
        return best_preps['barcode_encoding'].values[0], best_preps['barcode_processing'].values[0]

    # plain ol' progress log
    def log(self, val_idx, max_idx, times, total, idx, bar_prep, bar_enc, mode):
        estimated = sum(times) / len(times) * (total - idx)
        progress = round((idx / total) * 100, 2)
        with open(f"{self.results_dir}/log_{self.run_idx}.txt", 'a+') as f:
            f.write(
                f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | {mode}-{bar_prep}-{bar_enc} | sample: {val_idx} ({val_idx}/{max_idx}|{idx}/{total}) | progress: {progress}% |  ETA: {str(timedelta(seconds=estimated))}\n')

    # take the complete dataset add column with information about
    # a) LOOCV validation samples as we use a subset to speed things up and have balanced evaluation
    # b) k-fold CV validation folds (i.e. indices)
    def prepare_cv_ds(self):
        base_f = f'{self.data_dir}/records/merged/{self.job_id}_{self.marker}'
        if os.path.isfile(f'{base_f}_cv{self.cv_folds}.tsv'):
            return pd.read_csv(f'{base_f}_cv{self.cv_folds}.tsv', header=0, sep='\t')

        records = pd.read_csv(f'{base_f}.tsv', header=0, sep='\t')

        if self.cv_folds == 0:
            if self.subset:
                # this is LOOCV; we take 4 samples per class for separate models + validations
                records_shuffled = records.sample(frac=1, random_state=42).copy()
                records_subset = records_shuffled.groupby(['species_name']).head(4)

                records.loc[:, 'fold'] = 1  # train
                records.loc[records['record_id'].isin(records_subset['record_id']), 'fold'] = 0  # validation subset
            else:
                records.loc[:, 'fold'] = 0  # validate on every sample (individually)

            records['val_idx'] = records.index
        else:
            # we use stratified k-fold cross-validation
            # i.e. we make sure that class distribution stays the same across folds
            skf = StratifiedKFold(n_splits=self.cv_folds)
            for i, fold_indices in enumerate(skf.split(np.zeros(len(records)), records['species_name'])):
                records.loc[fold_indices[1], 'fold'] = i

        records.to_csv(f'{base_f}_cv{self.cv_folds}.tsv', header=True,
                       index=False, sep='\t')
        return records

    # split samples (for LOOCV) or folds (for k-fold CV) into batches according to parallel runs
    # iterate over either samples or folds and start training
    def train_samples(self, bar_enc, bar_prep, pretrainings, classifiers, mode='fused'):
        records = self.prepare_cv_ds()
        indices_to_do = self.batch_indices(self.records_todo(records, bar_enc, bar_prep, mode))

        if len(indices_to_do) == 0:
            return

        print(f'Training: {mode} with {bar_prep}-{bar_enc}')

        with open(f"{self.results_dir}/log_{self.run_idx}.txt", 'a+') as f:
            f.write(
                f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Boundaries -- {min(indices_to_do)} to {max(indices_to_do)} (total: {len(indices_to_do)} samples)\n')

        times = []
        for i, val_idx in enumerate(indices_to_do):
            sample_start = time.time()
            self.train_models(bar_enc, bar_prep, pretrainings, classifiers, mode, val_idx)
            sample_end = time.time()
            times.append(sample_end - sample_start)
            self.log(val_idx, max(indices_to_do), times, len(indices_to_do), i + 1, bar_prep, bar_enc, mode)

    # determines which samples/folds still need to be used a validation samples/folds
    def records_todo(self, records, bar_enc, bar_prep, mode='fused'):
        if self.cv_folds == 0:
            indices_to_do = records.loc[records['fold'] == 0, 'val_idx'].values
        else:
            indices_to_do = records['fold'].unique()

        res_fs = glob(f'{self.results_dir}/results*')

        # get already finished indices from result files
        if len(res_fs) == 0:
            finished_indices = []
        else:
            results = pd.concat([pd.read_csv(result_tsv, header=0, sep='\t') for result_tsv in res_fs])
            results = results[(results['barcode_processing'] == bar_prep) & (results['barcode_encoding'] == bar_enc)]
            if classifiers[0] != 'sep':
                results = results[results['round_2'].isin(classifiers)]
                min_records = len(classifiers)
            else:
                results = results[results['round_2'] == mode]
                min_records = 1
            results['finished_models'] = results.groupby(['val_idx'])['val_idx'].transform('count')
            finished_indices = results[results['finished_models'] >= min_records]['val_idx'].unique()

        unfinished_indices = [val_idx for val_idx in indices_to_do if val_idx not in finished_indices]
        return unfinished_indices

    # splits the models that still need to run into batches so multiple models can be trained in parallel
    def batch_indices(self, indices_to_do):
        subset_size = len(indices_to_do) // self.runs
        range_start = subset_size * self.run_idx
        range_end = len(indices_to_do) if self.runs == (self.run_idx + 1) else (self.run_idx + 1) * subset_size
        batched_indices = indices_to_do[range_start:range_end]
        return batched_indices
