import matplotlib.image as mpimg
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import UnidentifiedImageError
import PIL.Image
import re
import subprocess
import time
from tqdm import tqdm
from typing import Union
import warnings


class Commons:
    def __init__(self, project_dir, job_id, num_processes):
        self.num_processes = num_processes
        self.job_id = job_id

        # folders setup
        self.project_dir = project_dir
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.dir_base = f'{project_dir}/data/{job_id.replace(" ", "_")}'
        self.dir_merged_recs = os.path.join(self.dir_base, 'records', 'merged')
        self.dir_bold = os.path.join(self.dir_base, 'records', 'bold')
        self.dir_genbank = os.path.join(self.dir_base, 'records', 'genbank')
        self.dir_barcodes = os.path.join(self.dir_base, 'barcodes')
        self.dir_images = os.path.join(self.dir_base, 'images')
        self.dir_server = os.path.join(self.dir_base, 'server')

        self.marker: str = str()

    def mkdirs(self):
        for d in [self.dir_bold, self.dir_genbank,
                  self.dir_barcodes, self.dir_images,
                  self.dir_server, self.dir_merged_recs]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def log(self, msg: str, to_terminal: bool = False):
        with open(f'{self.dir_base}/log.txt', 'a') as f:
            f.write(f"{time.ctime()} - {msg}\n")

        if to_terminal:
            print(msg)

    # add image count to dataframe; used by >1 class
    def count_images(self, records, with_dup=True):
        if not with_dup:
            records['duplicate'] = 'False'

        records['image_cnt'] = records.apply(
            lambda x: len(x['image_urls'].split('|')) - x['duplicate'].count('True')
            if isinstance(x['image_urls'], str) and isinstance(x['duplicate'], str) else 0,
            axis=1)

        if not with_dup:
            records.drop(columns=['duplicate'], inplace=True)

        return records

    # wrapper to download all images of a given species
    def download_species_images(self, records: pd.DataFrame, image_column: str = 'image_urls'):
        species = records['species_name'].values[0]
        Path(f"{self.dir_images}/{self.marker}/{species}").mkdir(parents=True, exist_ok=True)

        records[['downloaded', 'image_path']] = records.apply(lambda row: self.download_image(row, image_column),
                                                              axis=1, result_type='expand')

        return records

    # download all images of the given dataset
    def download_images(self, records: pd.DataFrame):
        if 'downloaded' not in records.columns:
            self.log("Downloading images...")
            records.loc[:, 'downloaded'] = records.apply(
                lambda row: '|'.join(['False'] * len(row['image_urls'].split('|'))),
                axis=1
            )

            all_species = records['species_name'].unique()
            species_recs = [records.loc[records['species_name'] == species_name, :].copy()
                            for species_name in all_species]

            with mp.Pool(self.num_processes) as pool:
                new_records = list(tqdm(pool.imap(self.download_species_images,
                                                  species_recs),
                                        total=len(all_species)))

            records = pd.concat(new_records, ignore_index=True)
            records.sort_values('record_id', inplace=True)
            records.to_csv(f"{self.dir_merged_recs}/{self.job_id}_{self.marker}.tsv", sep='\t', header=True,
                           index=False)
        return records

    # download image of dataset row
    def download_image(self, image_info: pd.Series, image_column: str = 'image_urls') -> tuple:
        urls = image_info[image_column]
        if urls == 'FLORA':
            return 'True', image_info['image_path']

        species = image_info['species_name']
        record_id = image_info['record_id']
        was_downloaded = []
        file_path = None
        downloaded = False

        for image_url in urls.split('|'):
            if downloaded is True:
                # we just need one successful download per record
                was_downloaded.append(False)
                continue

            try:
                image_extension = re.search(r'(?<=\.)(jpe?g)|(png)|(tiff)', image_url).group(0)
            except:
                image_extension = 'jpg'
            image_file = f"{self.dir_images}/{self.marker}/{species}/{record_id}.{image_extension}"

            try:
                subprocess.run([
                    'wget',
                    '-O',
                    image_file,
                    image_url
                ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10
                )
                downloaded = self.check_image(image_file)
            except subprocess.TimeoutExpired:
                Path(image_file).unlink(missing_ok=True)
                downloaded = False

            was_downloaded.append(downloaded)
            if downloaded:
                file_path = image_file

        return '|'.join(map(str, was_downloaded)), file_path

    # check if image was downloaded correctly (i.e., file exists & can be read)
    def check_image(self, img_f: Union[Path, str]) -> bool:
        warnings.simplefilter('error')

        if not os.path.isfile(img_f):
            return False

        try:
            img = mpimg.imread(img_f)
            if img is None:
                raise TypeError
            return True
        except (TypeError, UserWarning, UnidentifiedImageError, OSError, PIL.Image.DecompressionBombError):
            Path(img_f).unlink(missing_ok=True)
            return False

    # updates rows with new images/image information
    def add_records_w_new_imgs(self, num_rows_to_pad: int, records_with_new_urls: pd.DataFrame,
                               records_wo_images: pd.DataFrame, records_w_images: pd.DataFrame,
                               image_column: str = 'image_urls') -> pd.DataFrame:
        # create rows with NaNs to be able to update original dataframe later on
        relevant_columns = [image_column, 'authors', 'license', 'gbif_observation_key']
        row_padding = pd.DataFrame.from_records([[None] * 4] * num_rows_to_pad,
                                                columns=relevant_columns)

        # merge dataframe with Nones with URLs to merge
        records_with_new_urls = pd.concat([records_with_new_urls.loc[:, relevant_columns], row_padding])

        if 'duplicate' in records_w_images.columns:
            # add duplicate = FALSE for records that now have images
            records_with_new_urls['duplicate'] = np.nan
            records_with_new_urls.loc[records_with_new_urls[image_column].notnull(), 'duplicate'] = False

        # update subset of records (i.e., without images) with padded image dataframe
        records_wo_images.drop(columns=relevant_columns + ['duplicate'], inplace=True, errors='ignore')

        records_w_new_images = records_wo_images.reset_index(drop=True).join(
            records_with_new_urls.reset_index(drop=True), how='left')

        column_types = {'related_data': bool}
        if 'duplicate' in records_w_images.columns:
            column_types['duplicate'] = bool

        records_w_images = records_w_images.astype(column_types)
        records_w_new_images = records_w_new_images.astype(column_types)

        if len(records_w_new_images) > 1 and len(records_w_images) > 1:
            # merge with subset of records that already had images
            records = pd.concat([records_w_images, records_w_new_images], ignore_index=True)
            return records
        elif len(records_w_new_images) > 1:
            return records_w_new_images
        else:
            return records_w_images

    # find duplicate images and update image information in dataset accordingly
    def mark_duplicates_by_image_for_dir(self, species_and_recs: tuple[str, pd.DataFrame]) -> pd.DataFrame:
        records = species_and_recs[1]

        species_image_files = records[records['image_path'].notnull()]['image_path'].values
        img_list = []

        for species_image_file in species_image_files:
            try:
                image_file_size = os.path.getsize(species_image_file)
                matched_images = [img for img_f_size, img in img_list if image_file_size == img_f_size]
                species_image = mpimg.imread(species_image_file)

                if len(matched_images) > 0:
                    seen_image = np.any(
                        [(species_image == image).all() if species_image.shape == image.shape else False for image in
                         matched_images])
                else:
                    seen_image = False
                img_already_processed = False
            except IOError:
                seen_image = False
                img_already_processed = True

            if seen_image or img_already_processed:
                img_record_info = Path(species_image_file).stem.split('_')[0]
                duplicate_info = str(
                    records[records['record_id'] == img_record_info]['duplicate'].values[0]).split(
                    '|')
                duplicate_info = ['True'] * len(duplicate_info)
                records.loc[records['record_id'] == img_record_info, 'duplicate'] = '|'.join(duplicate_info)

                Path(species_image_file).unlink(missing_ok=True)
            else:
                img_list.append([image_file_size, species_image])

        return records
