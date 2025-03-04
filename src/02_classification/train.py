import sys

import copy
from datetime import datetime
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
import time
import torch
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from typing import Union

from barcodejpg_dataset import BarcodeJPGDataset
from early_stopping import EarlyStopping
from model_bar_resnet import BarcodeJPGBar
from model_bar_resnet_sequential_embedding import SequentialEmbeddingBarcodeJPGBar
from model_barimg import BarcodeJPGFuse
from model_img import BarcodeJPGImg


class BarcodeJPGTrain:
    def __init__(self, job_id: str,
                 results_dir: Union[Path, str],
                 data_dir: Union[Path, str],
                 run_idx: int,
                 bar_encoding: str,
                 bar_processing: str,
                 classifiers: list[str],
                 pretrainings: list[str],
                 marker: str,
                 results: pd.DataFrame,
                 cv_folds: int = None,
                 batch_size: int = 32,
                 cv: bool = False,
                 dry: bool = False,
                 feature_extract: bool = True,
                 mode: str = 'fused',
                 num_epochs: int = 500,
                 num_workers: int = 4,
                 val_idx: int = None,
                 hide_progress: bool = False,
                 root_dir: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            raise Exception("No GPU")

        self.job_id = job_id
        self.run_idx = run_idx
        self.bar_processing = bar_processing
        self.bar_encoding = bar_encoding
        self.classifiers = classifiers
        self.pretrainings = pretrainings
        self.results = results
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.root_dir = root_dir
        if cv:
            self.model_results_dir = f'{self.results_dir}/{val_idx}/{bar_processing}/{bar_encoding}'
            df_f = f'{data_dir}/records/merged/{self.job_id}_{marker}_cv{cv_folds}.tsv'
        else:
            self.model_results_dir = f'{self.results_dir}/{bar_processing}/{bar_encoding}'
            df_f = f'{data_dir}/records/merged/{self.job_id}_{marker}.tsv'
        self.log_file = f"{self.model_results_dir}/log.txt"
        self.batch_log_file = f"{self.results_dir}/log_batch_{self.run_idx}.tsv"
        self.wip_file = f"{self.results_dir}/wips_{self.run_idx}.tsv"
        self.result_file = f"{self.results_dir}/results_{self.run_idx}.tsv"

        self.batch_size: int = batch_size
        self.cv: bool = cv
        self.cv_folds: int = cv_folds
        self.dry: bool = dry
        self.feature_extract: bool = feature_extract
        self.mode = mode
        self.num_epochs: int = num_epochs
        self.val_idx: int = val_idx
        self.out_features: int = 0
        self.hide_progress: bool = hide_progress

        self.dataloaders, self.num_classes = self.gen_dataloaders(data_dir, df_f, num_workers)

    # plain ol' progress log part 2 ;)
    def log(self, msg: str, command_line: bool = False):
        with open(self.log_file, 'a+') as f:
            f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: {msg}\n')
        if command_line:
            print(msg)

    # checks whether training has been started by sibling run
    def is_wip(self, round_1: str, round_2: str, bar_prep: str, bar_enc: str) -> bool:
        wip = False

        result_files = glob(f"{self.results_dir}/wips_*.tsv")
        if len(result_files) > 0:
            results = pd.concat([pd.read_csv(result_file, header=0, sep='\t') for result_file in result_files])

            res = results.loc[(results['val_idx'] == self.val_idx) &
                              (results['barcode_processing'] == bar_prep) &
                              (results['barcode_encoding'] == bar_enc) &
                              (results['round_1'] == round_1) & (results['round_2'] == round_2), :]

            wip = len(res) > 0

        if not wip:
            if os.path.isfile(self.wip_file):
                wips = pd.read_csv(self.wip_file, sep='\t', header=0)
            else:
                wips = pd.DataFrame(columns=['barcode_processing', 'barcode_encoding',
                                             'round_1', 'round_2', 'val_idx'])

            wips.loc[len(wips)] = [self.bar_processing, self.bar_encoding, round_1, round_2, self.val_idx]
            wips.to_csv(self.wip_file, header=True, index=False, sep='\t')

        return wip

    # checks whether training has been finished already
    def is_done(self, round_1: str, round_2: str) -> bool:
        done = False

        if round_2 == 'img':
            bar_enc = 'one_hot_bar'
            bar_prep = 'aligned_barcode'
        else:
            bar_enc = self.bar_encoding
            bar_prep = self.bar_processing

        result_files = glob(f"{self.results_dir}/results_*.tsv")
        if len(result_files) > 0:
            results = pd.concat([pd.read_csv(result_file, header=0, sep='\t') for result_file in result_files])

            res = results.loc[(results['barcode_processing'] == bar_prep) &
                              (results['barcode_encoding'] == bar_enc) &
                              (results['round_1'] == round_1) & 
                              (results['round_2'] == round_2), :]

            if self.val_idx:
                res = res.loc[res['val_idx'] == self.val_idx, :]

            done = len(res) > 0

        if done:
            return done

        return self.is_wip(round_1, round_2, bar_prep, bar_enc)

    # generates dataloaders for training
    def gen_dataloaders(self, data_dir, df_f, num_workers) -> tuple[dict, int]:
        barcode_jpg_datasets = {phase: BarcodeJPGDataset(self.job_id, f'{data_dir}/server', phase,
                                                         self.bar_processing, df_f, self.mode,
                                                         self.cv, self.cv_folds, val_idx=self.val_idx,
                                                         dry=self.dry)
                                for phase in ['train', 'val']}
        num_classes = len(barcode_jpg_datasets['train'].class_map)

        dataloaders = {phase: DataLoader(barcode_jpg_datasets[phase], batch_size=self.batch_size, shuffle=True,
                                         num_workers=num_workers)
                       for phase in ['train', 'val']}

        if self.val_idx == 0 or self.val_idx is None:
            self.print_class_map(barcode_jpg_datasets['train'].class_map)
        return dataloaders, num_classes

    # print information about classes-to-indices only for first of multiple parallel runs or if only run
    def print_class_map(self, class_map: dict):
        class_map = pd.DataFrame.from_dict(class_map, orient='index', columns=['idx'])
        class_map['species_name'] = class_map.index
        class_map.to_csv(f'{self.results_dir}/class_map.tsv', sep='\t', header=True,
                         index=False)

    # actual training of model
    def train_model(self,
                    model: BarcodeJPGFuse,
                    criterion: torch.nn,
                    optimizer: torch.optim,
                    train_mode: str = 'fused',
                    slf: bool = False) -> tuple:
        since = time.time()

        history = pd.DataFrame(columns=['epoch', 'phase', 'acc', 'loss'])
        predictions = pd.DataFrame(columns=['epoch', 'sample_idx', 'Y_gt', 'Y_pred', 'score', 'gt_score'])
        bar_preds = pd.DataFrame(columns=['epoch', 'sample_idx', 'Y_gt', 'Y_pred', 'score', 'gt_score'])
        img_preds = pd.DataFrame(columns=['epoch', 'sample_idx', 'Y_gt', 'Y_pred', 'score', 'gt_score'])

        if self.dry:
            return model, optimizer, history, 0.0, predictions, bar_preds, img_preds, 0, 0

        model_state_dict = copy.deepcopy(model.state_dict())
        optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
        best_acc = 0.0
        best_epoch = 0
        best_loss = float('inf')
        early_stopper = EarlyStopping(patience=20)

        for epoch in range(self.num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()  # only evaluate during validation

                running_loss = 0.0
                running_corrects = 0.0

                # iterate over data
                with tqdm(self.dataloaders[phase], unit="batch",
                          disable=self.hide_progress or phase == 'val') as tepoch:
                    for barcodes, images, labels, sample_idxs in tepoch:
                        if self.mode != 'img':
                            barcodes = barcodes.to(self.device)
                        if self.mode != 'bar':
                            images = images.to(self.device)
                        labels = labels.to(self.device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            # get model outputs and calculate loss
                            if train_mode == 'fused':
                                outputs, barcode_outputs, image_outputs = model(barcodes, images)
                            elif train_mode == 'bar':
                                outputs = model(barcodes)
                            else:
                                outputs = model(images)

                            loss = criterion(torch.log(outputs + 1e-20), labels)

                            scores, preds = torch.max(outputs, 1)
                            gt_scores = torch.gather(outputs, 1, labels.view(-1, 1))

                            if slf:  # get predictions from individual models in addition to fused preds
                                barcode_scores, barcode_preds = torch.max(barcode_outputs, 1)
                                barcode_gt_scores = torch.gather(barcode_outputs, 1, labels.view(-1, 1))
                                image_scores, image_preds = torch.max(image_outputs, 1)
                                image_gt_scores = torch.gather(image_outputs, 1, labels.view(-1, 1))

                            # backpropagation
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * barcodes.size(0)
                        correct = torch.sum(preds.eq(labels.data))
                        running_corrects += correct
                        if phase == 'train':
                            tepoch.set_postfix(loss=loss.item(), accuracy=f'{100. * (correct / self.batch_size)}%')
                        else:
                            # update training information dataframes
                            labels_cpu = labels.data.cpu()
                            preds_cpu = preds.cpu()
                            scores_cpu = scores.cpu()
                            gt_scores_cpu = gt_scores.cpu()
                            if slf:
                                barcode_preds_cpu = barcode_preds.cpu()
                                barcode_scores_cpu = barcode_scores.cpu()
                                barcode_gt_scores_cpu = barcode_gt_scores.cpu()

                                image_preds_cpu = image_preds.cpu()
                                image_scores_cpu = image_scores.cpu()
                                image_gt_scores_cpu = image_gt_scores.cpu()

                            for i, sample_idx in enumerate(sample_idxs):
                                predictions.loc[len(predictions)] = [epoch, sample_idx, labels_cpu[i].item(),
                                                                     preds_cpu[i].item(), scores_cpu[i].item(),
                                                                     gt_scores_cpu[i].item()]
                                if slf:
                                    bar_preds.loc[len(bar_preds)] = [epoch, sample_idx, labels_cpu[i].item(),
                                                                     barcode_preds_cpu[i].item(),
                                                                     barcode_scores_cpu[i].item(),
                                                                     barcode_gt_scores_cpu[i].item()]
                                    img_preds.loc[len(img_preds)] = [epoch, sample_idx, labels_cpu[i].item(),
                                                                     image_preds_cpu[i].item(),
                                                                     image_scores_cpu[i].item(),
                                                                     image_gt_scores_cpu[i].item()]

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                if np.isnan(epoch_loss):
                    raise Exception('Loss in NaN. Please check.')
                epoch_acc = running_corrects / len(self.dataloaders[phase].dataset)

                self.log('Epoch: {} - {} Loss: {:.4f} Acc: {:.4f}%'.format(epoch, phase, epoch_loss, 100. * epoch_acc),
                         command_line=False)
                if phase == 'val':
                    early_stopper(val_loss=epoch_loss)

                    # deep copy the model if it's best model so far
                    if early_stopper.best_epoch:
                        best_acc = epoch_acc
                        best_epoch = epoch
                        best_loss = epoch_loss
                        model_state_dict = copy.deepcopy(model.state_dict())
                        optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

                history.loc[len(history)] = [epoch, phase, float(epoch_acc), float(epoch_loss)]

            if early_stopper.early_stop:
                self.log(f'Early stopping after epoch: {epoch}')
                break

        time_elapsed = time.time() - since
        self.log('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.log(f'Best epoch: {best_epoch}')
        self.log('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        return model, optimizer, history, float(best_acc), predictions, bar_preds, img_preds, best_epoch, best_loss

    # collects parameters to be trained and sets up the optimizer
    def setup_params(
            self,
            model: BarcodeJPGFuse, descr: str, log: bool = True
    ) -> tuple[Union[BarcodeJPGFuse, BarcodeJPGBar, BarcodeJPGImg, SequentialEmbeddingBarcodeJPGBar], torch.optim, int]:
        # send model to GPU
        model = model.to(self.device)

        if log:
            self.log(f"Params to learn {descr}:", command_line=False)
        params_to_update = []
        trainable_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                trainable_params += torch.numel(param)
                if log:
                    self.log(f"\t{name}", command_line=False)

        self.log(f"Total trainable parameters: {trainable_params}", command_line=False)
        # Observe that all parameters are being optimized
        optimizer_ft = torch.optim.Adam(params_to_update, lr=0.0001)
        return model, optimizer_ft, trainable_params

    # initializes barcode and image models
    def setup_base_models(self, classifier: str
                          ) -> tuple[Union[BarcodeJPGBar, SequentialEmbeddingBarcodeJPGBar], BarcodeJPGImg]:
        w_classifier = False if classifier == 'dense_mid' or classifier == 'rf' else True
        img_model_inst = BarcodeJPGImg(self.num_classes, classifier_mode=classifier, w_classifier=w_classifier)
        if self.bar_encoding == 'one_hot_bar':
            bar_model = BarcodeJPGBar(num_classes=self.num_classes, classifier_mode=classifier,
                                      w_classifier=w_classifier)
        else:
            bar_model = SequentialEmbeddingBarcodeJPGBar(classifier_mode=classifier, num_classes=self.num_classes,
                                                         w_classifier=w_classifier)

        bar_model = bar_model.to(self.device)
        img_model_inst.model = img_model_inst.model.to(self.device)

        return bar_model, img_model_inst

    # save model and optimizer states as well as epoch history and predictions
    def save_model_assets(self, model, optimizer, history, preds, round_idx: int, classifier, out_dir,
                          bar_preds=None, img_preds=None):
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(state, f"{out_dir}/model_state_round{round_idx}_{classifier}.pickle", pickle_protocol=3)
        history.to_csv(f"{out_dir}/history_round{round_idx}_{classifier}.tsv", sep='\t', header=True, index=False)
        preds.to_csv(f"{out_dir}/preds_round{round_idx}_{classifier}.tsv", sep='\t', header=True, index=False)

        try:
            bar_preds.to_csv(f"{out_dir}/bar_preds_round{round_idx}_{classifier}.tsv", sep='\t', header=True,
                             index=False)
            img_preds.to_csv(f"{out_dir}/img_preds_round{round_idx}_{classifier}.tsv", sep='\t', header=True,
                             index=False)
        except AttributeError:
            pass

        self.generate_history_plot(history, f"{out_dir}/history_round{round_idx}_{classifier}.svg")

    # generate plot depicting epoch accuracy + loss
    def generate_history_plot(self, history: pd.DataFrame, out_f: str):
        fig, axs = plt.subplots(nrows=2, ncols=1)
        history.set_index('epoch', inplace=True)
        history.loc[:, ['phase', 'acc']].groupby('phase')['acc'].plot(x='epoch', legend=True,
                                                                      ax=axs[0], title='Accuracy')
        history.loc[:, ['phase', 'loss']].groupby('phase')['loss'].plot(x='epoch', legend=True,
                                                                        ax=axs[1], title='Loss')
        fig.tight_layout()
        fig.savefig(out_f, format='svg')

    # log number of parameters
    def write_params(self, training_mode: str, classifier: str, bar_ftrs: int, img_ftrs: int,
                     classifier_params: int, shared_params: int, total_params: int, trainable_params: int):
        param_stats_f = f"{self.results_dir}/param_stats.tsv"

        if os.path.isfile(param_stats_f):
            param_stats = pd.read_csv(param_stats_f, header=0, sep='\t')
        else:
            param_stats = pd.DataFrame(columns=['barcode_processing', 'barcode_encoding', 'round_1',
                                                'round_2', 'bar_ftrs', 'img_ftrs', 'mlp', 'shared', 'total',
                                                'trainable'])

        param_stats.loc[len(param_stats)] = [self.bar_processing,
                                             self.bar_encoding,
                                             training_mode, classifier,
                                             bar_ftrs, img_ftrs,
                                             classifier_params, shared_params,
                                             total_params, trainable_params]
        param_stats.to_csv(param_stats_f, sep='\t', header=True, index=False)

    # wrapper function; iterates over first and second trainings specified by the user
    def run(self):
        for pretraining in self.pretrainings:  # first training
            for classifier in self.classifiers:  # second training with first as its starting point
                out_dir = f"{self.model_results_dir}/tr1_{pretraining}"
                Path(out_dir).mkdir(parents=True, exist_ok=True)
    
                loss_fxn = torch.nn.NLLLoss()
                bar_model, img_model_inst = self.setup_base_models(pretraining)
    
                if pretraining == 'sep':
                    bar_model, img_model_inst = self.separate_training(out_dir, loss_fxn, bar_model, img_model_inst)
                elif 'score' in pretraining:
                    bar_model, img_model_inst = self.score_level_training(1, 'slf', pretraining, out_dir, loss_fxn,
                                                                          bar_model=bar_model,
                                                                          img_model_inst=img_model_inst)
                elif pretraining == 'dense_mid':
                    bar_model, img_model_inst = self.dense_training(1, 'dense_mid', pretraining, out_dir, loss_fxn, bar_model,
                                                                    img_model_inst)
                else:
                    bar_model, img_model_inst = self.dense_training(1, 'dense_late', pretraining, out_dir, loss_fxn, bar_model,
                                                                    img_model_inst)

                if classifier == pretraining:
                    continue

                if 'score' in classifier:
                    self.score_level_training(2, pretraining, classifier, out_dir, loss_fxn, bar_model=bar_model, img_model_inst=img_model_inst)
                elif classifier == 'dense_mid' or classifier == 'dense_late':
                    self.dense_training(2, pretraining, classifier, out_dir, loss_fxn, bar_model, img_model_inst)
                else:
                    self.rf_training(pretraining, out_dir, bar_model, img_model_inst)

    # separate training for barcodes and images
    def separate_training(self, out_dir: str, loss_fxn: torch.nn,
                          bar_model: Union[BarcodeJPGBar, SequentialEmbeddingBarcodeJPGBar],
                          img_model_inst: BarcodeJPGImg) -> list:
        models = []
        out_features = {}

        for model, abbr in [[bar_model, 'bar'], [img_model_inst.model, 'img']]:
            if self.mode == 'fused' or self.mode == abbr:
                if self.is_done('sep', abbr):
                    model = self.load_done_model('round 1', f'sep:{abbr}', abbr, out_dir,
                                                 model)
                else:
                    model = self.train_model_wrapper('round 1', 'sep', abbr, out_dir,
                                                     abbr, model, loss_fxn, img_model_inst=img_model_inst)

            if abbr == 'img':
                img_model_inst.model = model
                models.append(img_model_inst)
            else:
                models.append(model)
            out_features[abbr] = model.fc[0].in_features

        self.out_features = sum(out_features.values())
        return models

    # trains with either max, product or sum score fusion
    def score_level_training(self, train_round: int, pretraining: str, slf_mode: str, out_dir: str,
                             loss_fxn: torch.nn, bar_model: Union[BarcodeJPGBar, SequentialEmbeddingBarcodeJPGBar],
                             img_model_inst: BarcodeJPGImg) -> BarcodeJPGFuse:
        if train_round == 2:
            bar_model, img_model_inst = self.update_sep_models(bar_model, img_model_inst, True, slf_mode)

        model = BarcodeJPGFuse(self.job_id, bar_model, img_model_inst.model, classifier_mode='slf',
                               slf_mode=slf_mode,
                               num_classes=self.num_classes, trainable=True)

        if not self.is_done(pretraining, slf_mode):
            model = self.train_model_wrapper(f'round {train_round}',
                                             pretraining, slf_mode, out_dir,
                                             'fused', model, loss_fxn, slf=True)
        elif train_round != 2:
            model = self.load_done_model(f'round {train_round}',
                                         f'{pretraining}:{slf_mode}',
                                         slf_mode, out_dir,
                                         model)
        return model

    # trains with dense-based fusion either directly after last convolutional layer or after first dense layer
    def dense_training(self, train_round: int, pretraining: str, dense_mode: str, out_dir: str, loss_fxn: torch.nn,
                       bar_model: Union[BarcodeJPGBar, SequentialEmbeddingBarcodeJPGBar],
                       img_model_inst: BarcodeJPGImg) -> BarcodeJPGFuse:
        if train_round == 2:
            bar_model, img_model_inst = self.update_sep_models(bar_model, img_model_inst, 
                                                               dense_mode == 'dense_late', dense_mode)
        if dense_mode == 'dense_mid':
            input_ftrs = self.out_features
        else:
            input_ftrs = None
            # chop off last layer of regular classifiers
            bar_model.fc = torch.nn.Sequential(*(list(bar_model.fc.children())[:-2]))
            img_model_inst.model.fc = torch.nn.Sequential(*(list(img_model_inst.model.fc.children())[:-2]))
        model = BarcodeJPGFuse(self.job_id, bar_model, img_model_inst.model, classifier_mode=dense_mode,
                               slf_mode=None,
                               num_classes=self.num_classes, trainable=True,
                               input_ftrs=input_ftrs)

        if not self.is_done(pretraining, dense_mode):
            model = self.train_model_wrapper(f'round {train_round}',
                                             pretraining, dense_mode, out_dir,
                                             'fused', model, loss_fxn)
        elif train_round != 2:
            model = self.load_done_model(f'round {train_round}',
                                         f'{pretraining}:{dense_mode}',
                                         dense_mode, out_dir, model)

        return model

    # trains with random forest classifier directly after last convolutional layer
    def rf_training(self, pretraining: str, out_dir: str, bar_model: Union[BarcodeJPGBar, SequentialEmbeddingBarcodeJPGBar],
                    img_model_inst: BarcodeJPGImg):

        if self.is_done(pretraining, 'rf'):
            return
        
        bar_model, img_model_inst = self.update_sep_models(bar_model, img_model_inst, False, 'rf')
        img_model = img_model_inst.model

        rf_grid_search = pd.read_csv(f'{self.root_dir}/results/rf_grid_search_results.tsv', header=0, sep='\t')
        rf_grid_search = rf_grid_search.loc[rf_grid_search['job_id'] == self.job_id, :]

        try:
            max_depth = int(rf_grid_search['max_depth'].values[0])
        except ValueError:
            max_depth = None
            
        n_estimators = rf_grid_search['n_estimators'].values[0]
        min_samples_split = rf_grid_search['min_samples_split'].values[0]
        min_samples_leaf = rf_grid_search['min_samples_leaf'].values[0]
        max_features = rf_grid_search['max_features'].values[0]
        bootstrap = rf_grid_search['bootstrap'].values[0]
        
        rfc = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                     max_features=max_features, bootstrap=bootstrap)
        
        since = time.time()
        predictions = pd.DataFrame(columns=['sample_idx', 'Y_gt', 'Y_pred'])
        accuracy = None

        for phase in ['train', 'val']:
            outputs_fused_np = np.empty(shape=[len(self.dataloaders[phase].dataset), 4096])
            labels_np = np.empty(shape=[len(self.dataloaders[phase].dataset)])
            sample_idx_total = []
            
            for i, batch_data in enumerate(self.dataloaders[phase]):
                barcodes, images, labels, sample_idxs = batch_data[0], batch_data[1], batch_data[2], batch_data[3]
                sample_idx_total.extend(sample_idxs)

                barcodes = barcodes.to(self.device)
                images = images.to(self.device)

                with torch.no_grad():
                    bar_model.eval()
                    img_model.eval()

                    outputs_bar = bar_model(barcodes)
                    outputs_img = img_model(images)

                    outputs_fused_np[i * self.batch_size:(i + 1) * self.batch_size] = torch.cat(
                        (outputs_bar, outputs_img), dim=1).cpu().numpy()
                    labels_np[i * self.batch_size:(i + 1) * self.batch_size] = labels.data

            if phase == 'train':
                rfc.fit(outputs_fused_np, labels_np)
            else:
                preds = rfc.predict(outputs_fused_np)
                for i, sample_idx in enumerate(sample_idx_total):
                    predictions.loc[len(predictions)] = [sample_idx, labels_np[i],
                                                         preds[i]]
                correct = sum(preds == labels_np)
                accuracy = correct / len(self.dataloaders[phase].dataset)

        time_elapsed = time.time() - since
        self.log('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.log('Best val Acc: {:4f}'.format(accuracy))

        pickle.dump(rfc, open(f"{out_dir}/model_state_round2_rf.pickle", 'wb'))
        predictions.to_csv(f"{out_dir}/preds_round2_rf.tsv", sep='\t', header=True, index=False)
        self.results.loc[len(self.results)] = [self.bar_processing, self.bar_encoding,
                                               pretraining, 'rf',
                                               accuracy, None, None, self.val_idx]
        self.results.to_csv(self.result_file, sep='\t', header=True, index=False)

    # if model has already been trained, load finished model (only for training round 1 as we need model for round 2)
    def load_done_model(
            self, train_round: str, model_info: str, classifier: str, out_dir: str,
            model: Union[BarcodeJPGFuse, BarcodeJPGBar, models.resnet50, SequentialEmbeddingBarcodeJPGBar]
    ) -> Union[BarcodeJPGFuse, BarcodeJPGBar, models.resnet50, SequentialEmbeddingBarcodeJPGBar]:
        self.log(f"Skipping training for {train_round}: {model_info}")
        state = torch.load(f"{out_dir}/model_state_round1_{classifier}.pickle")
        model.load_state_dict(state['state_dict'])
        model, optimizer, trainable_params = self.setup_params(model, model_info, log=False)
        optimizer.load_state_dict(state['optimizer'])
        return model

    # update classifiers of separate models for round 2 according to new classifier/fusion
    def update_sep_models(self, bar_model, img_model_inst, w_classifier: bool, classifier: str
                          ) -> tuple[Union[BarcodeJPGBar, SequentialEmbeddingBarcodeJPGBar], BarcodeJPGImg]:
        bar_model.w_classifier = w_classifier
        img_model_inst.w_classifier = w_classifier

        if classifier == 'dense_mid' or classifier == 'rf':
            # remove classifiers of separate models
            bar_model.fc = torch.nn.Identity()
            img_model_inst.model.fc = torch.nn.Identity()
        else:
            # replace classifiers of separate models
            bar_model.fc = bar_model.get_classifier(classifier)
            img_model_inst.model.fc = img_model_inst.get_classifier(classifier)

        bar_model = bar_model.to(self.device)
        img_model_inst.model = img_model_inst.model.to(self.device)
        return bar_model, img_model_inst

    # wrapper function for training that sets up parameters, starts the training and saves the results/metadata
    def train_model_wrapper(
            self, train_round: str, pretraining: str, classifier: str, out_dir: str, train_mode: str,
            model: Union[BarcodeJPGFuse, BarcodeJPGBar, SequentialEmbeddingBarcodeJPGBar, models.resnet50],
            loss_fxn: torch.nn, slf: bool = False, img_model_inst: BarcodeJPGImg = None
    ) -> Union[BarcodeJPGFuse, BarcodeJPGBar, BarcodeJPGImg, SequentialEmbeddingBarcodeJPGBar]:
        self.log(f"Starting training for {train_round}: {pretraining}-{classifier}")
        model, optimizer, trainable_params = self.setup_params(model, f'{pretraining}-{classifier}')
        model, optimizer, history, best_acc, preds, bar_preds, img_preds, best_epoch, best_loss = self.train_model(
            model,
            loss_fxn,
            optimizer,
            train_mode=train_mode,
            slf=slf)
        self.results.loc[len(self.results)] = [self.bar_processing, self.bar_encoding,
                                               pretraining, classifier,
                                               best_acc, best_loss, best_epoch, self.val_idx]
        self.results.to_csv(self.result_file, sep='\t', header=True, index=False)
        train_round_idx = 1 if train_round == 'round 1' else 2
        self.save_model_assets(model, optimizer, history, preds, train_round_idx, classifier, out_dir, bar_preds, img_preds)

        try:
            model_params = model.count_model_params()
        except AttributeError:
            model_params = img_model_inst.count_model_params()
        self.write_params(pretraining, classifier, model_params[0],
                          model_params[1], model_params[2], model_params[3], model_params[4], trainable_params)

        return model
