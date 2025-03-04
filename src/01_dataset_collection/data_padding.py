import glob
from itertools import repeat
import multiprocessing as mp
import numpy as np
import os.path
import pandas as pd
import pandas.errors
from pathlib import Path
import re
import requests
from retry import retry
from tqdm import tqdm

from commons import Commons


class DataPadding(Commons):
    def __init__(self, project_dir, job_id, num_processes, fresh_images, marker):
        super().__init__(project_dir, job_id, num_processes)

        self.marker = marker
        self.fresh_images = fresh_images
        self.dir_urls = f"{self.dir_merged_recs}/urls"
        self.flora_dir = f"{self.project_dir}/data/Flora/{self.job_id}"

        if self.fresh_images:
            # inaturalist research grade observations
            self.gbif_basis = 'dataset_key=50c9509d-22c7-4a22-a47d-8c48425ef4a7&'
        else:
            self.gbif_basis = 'basisOfRecord=PRESERVED_SPECIMEN&basisOfRecord=MATERIAL_SAMPLE&'

    # add images from Flora and/or GBIF to records without BOLD images
    def add_species_imgs(self, args) -> pd.DataFrame:
        species_and_recs, ambiguous_urls = args
        species, records, recs_with_imgs = species_and_recs[0], species_and_recs[1], species_and_recs[2]

        if self.fresh_images:
            # add flora images
            records = self.add_flora_images(records, species)
        else:
            # copy URLs from unused records&markers
            records = self.mirror_urls_from_unused_recs(records, recs_with_imgs, ambiguous_urls)

        # for remaining entries search GBIF
        records = self.add_gbif_imgs(records, species)

        return records

    def add_flora_images(self, records: pd.DataFrame, species: str) -> pd.DataFrame:
        flora_images = self.flora_images(species)
        image_urls = (['FLORA'] * len(flora_images)) + ([np.nan] * len(records))
        m_urls = flora_images + ([np.nan] * len(records))
        records['image_urls'] = image_urls[:len(records)]
        records['image_path'] = m_urls[:len(records)]
        return records

    # search for Flora images
    def flora_images(self, species_name: str) -> list:
        species_dir = glob.glob(f"{self.flora_dir}/{species_name}*")
        if len(species_dir) > 0:
            return glob.glob(f"{species_dir[0]}/*_inflorescence*")
        else:
            return []

    # search for images via GBIF API and add them to dataset
    def add_gbif_imgs(self, records: pd.DataFrame, species: str, rely_on_saved_urls: bool = True,
                      image_column: str = 'image_urls') -> pd.DataFrame:
        try:
            marker_urls = records.deepcopy()
            marker_urls[image_column] = marker_urls[image_column].str.split('|')
            marker_urls = marker_urls.explode(image_column)
            marker_urls = marker_urls.loc[:, [image_column, 'authors', 'license', 'gbif_observation_key']]
        except AttributeError:
            marker_urls = pd.DataFrame(columns=[image_column, 'authors', 'license', 'gbif_observation_key'])
        gbif_urls = self.search_gbif(species, marker_urls,
                                     len(records[records[image_column].isnull()]), rely_on_saved_urls,
                                     image_column=image_column)
        records = self.pad_and_add_urls(records, gbif_urls, image_column=image_column)
        return records

    @retry(requests.exceptions.ConnectionError, delay=1, tries=5)
    def search_gbif(self, species_name_og: str, urls_so_far: pd.DataFrame, max_records: int,
                    rely_on_saved_urls: bool, image_column: str = 'image_urls') -> pd.DataFrame:
        species_name = species_name_og.replace(' ', '+').replace('[', '').replace(']', '')
        species_name_f = species_name_og.replace('/', '_').replace('[', '').replace(']', '')
        url_dir = f"{self.dir_urls}/{species_name_f.split(' ')[0]}"
        Path(url_dir).mkdir(parents=True, exist_ok=True)
        species_f = f"{url_dir}/{species_name_f.replace(' ', '_')}.tsv"

        if os.path.isfile(species_f):
            try:
                saved_urls = pd.read_csv(species_f, header=0, sep='\t')
                if image_column == 'image_urls':
                    saved_urls.rename(columns={'image_url': 'image_urls'}, inplace=True)
            except pandas.errors.EmptyDataError:
                saved_urls = pd.DataFrame(columns=[image_column, 'authors', 'license', 'gbif_observation_key'])

            if rely_on_saved_urls:
                return saved_urls
            else:
                urls_so_far = pd.concat([saved_urls, urls_so_far]).drop_duplicates(subset=[image_column])

        new_urls = {}
        gbif_rec_limit = 300
        gbif_rec_cnt = max_records
        chunk_idx = 0

        while len(new_urls) < max_records and gbif_rec_limit * chunk_idx < gbif_rec_cnt:
            gbif_offset = gbif_rec_limit * chunk_idx
            gbif_url = f'https://api.gbif.org/v1/occurrence/search?{self.gbif_basis}scientificName={species_name}&limit={gbif_rec_limit}&offset={gbif_offset}'
            try:
                response = requests.get(gbif_url, timeout=10)
                gbif_out = response.json()
                gbif_rec_cnt = int(gbif_out['count'])
                parsed_urls = self.parse_gbif(gbif_out, num_records=max_records)
                if len(parsed_urls) > 0:
                    new_urls = parsed_urls | new_urls
            except (ValueError, requests.exceptions.ChunkedEncodingError):
                self.log(f"Could not download images from: {gbif_url}")
            chunk_idx += 1

        if len(new_urls) > 0:
            # remove URLs that are duplicates of already in use URLs
            new_urls = {new_url: metadata for (new_url, metadata) in new_urls.items() if new_url not in urls_so_far}
            new_urls = pd.DataFrame.from_dict(new_urls, orient='index')
            new_urls[image_column] = new_urls.index
        else:
            new_urls = pd.DataFrame(columns=[image_column, 'authors', 'license', 'gbif_observation_key'])

        if image_column == 'image_urls':
            new_urls.rename(columns={'image_urls': 'image_url'}).to_csv(species_f, header=True, index=False, sep='\t')
        else:
            new_urls.to_csv(species_f, header=True, index=False, sep='\t')

        return new_urls

    # fetches URLs of species from unused records (i.e. from other markers and from other clusters)
    def mirror_urls_from_unused_recs(self, records: pd.DataFrame, other_recs: pd.DataFrame,
                                     ambiguous_urls: list) -> pd.DataFrame:
        try:
            in_use_urls = list(records['image_urls'].str.split('|').explode().unique())
        except AttributeError:
            in_use_urls = []

        other_recs['image_urls'] = other_recs['image_urls'].str.split('|')
        other_recs = other_recs.explode('image_urls')
        other_recs.drop_duplicates(subset=['record_id', 'image_urls'], inplace=True)

        records_with_urls_to_mirror = other_recs[~(other_recs['image_urls'].isin(in_use_urls) |
                                                   other_recs['image_urls'].isin(ambiguous_urls))].copy()

        records = self.pad_and_add_urls(records, records_with_urls_to_mirror)
        return records

    # splits records with images from records without images and adds as many images as were found
    def pad_and_add_urls(self, records: pd.DataFrame, records_with_new_urls: pd.DataFrame,
                         image_column: str = 'image_urls') -> pd.DataFrame:
        # get number of rows to pad with images
        idx = (records[image_column].isnull())
        records_wo_images = records.loc[idx, :].copy()
        records_w_images = records.loc[~idx, :].copy()

        # get number of rows that could not be padded
        num_rows_to_pad = len(records_wo_images) - len(
            records_with_new_urls[records_with_new_urls[image_column].isnull()])

        records = self.add_records_w_new_imgs(num_rows_to_pad, records_with_new_urls,
                                              records_wo_images, records_w_images,
                                              image_column=image_column)

        return records

    # wrapper function for addition of images via GBIF/Flora
    def add_images(self, records: pd.DataFrame) -> pd.DataFrame:
        # 0. check if records with related_data == False have images
        if max(records[~records['related_data']]['image_cnt']) != 0:
            self.log("Records already image-padded. Continuing...")
        else:
            self.log("Image-padding records...")
            self.log(f"Starting with {len(records)} entries...")
            # 1. basic info
            all_species = records['species_name'].unique()

            # 3. set up params for image addition per species
            total_records = pd.read_csv(f"{self.dir_merged_recs}/{self.job_id}.tsv", sep='\t', header=0)
            species_and_recs = [[species,
                                 records[records['species_name'] == species],
                                 total_records[(total_records['species_name'] == species) &
                                               (total_records['image_urls'].notnull())]]
                                for species in all_species]

            # get ambiguous URLs
            url_species_cnt = total_records.explode('image_urls')[['image_urls', 'species_name']].groupby('image_urls')[
                'species_name'].nunique()
            ambiguous_urls = list(url_species_cnt[url_species_cnt > 1].index.values)

            # with mp.Pool(self.num_processes) as pool:
            with mp.Pool(self.num_processes) as pool:
                new_records = list(tqdm(pool.imap(self.add_species_imgs,
                                                  zip(species_and_recs,
                                                      repeat(ambiguous_urls))),
                                        total=len(all_species)))

            records = pd.concat(new_records, ignore_index=True)

            # 5. kick out records w/o any images
            len_before = len(records)
            records.dropna(subset=['image_urls'], inplace=True)
            self.log(f"Removed {len_before - len(records)} entries without images...")
            self.log(f"Keeping {len(records)} entries with images...")

            # 6. set duplicate value for newly collected images to false and count images
            records = self.count_images(records)

            records.to_csv(f"{self.dir_merged_recs}/{self.job_id}_{self.marker}.tsv", sep='\t', header=True,
                           index=False)
        return records

    # checks if added images can be downloaded and are not duplicates of already included images
    def quality_check_padding(self, records: pd.DataFrame) -> pd.DataFrame:
        records = records.astype({'downloaded': 'bool', 'duplicate': 'bool'})
        len_before = len(records)

        records.loc[(records['image_url'].isnull()) |
                    ~(records['downloaded']) |
                    (records.duplicated(subset='image_url')) |
                    (records['duplicate']),
                    ['image_url', 'downloaded', 'duplicate']] = None, False, False

        for species in tqdm(records[records['image_url'].isnull()]['species_name'].unique()):
            species_recs_idx = records['species_name'] == species
            records[species_recs_idx] = self.add_gbif_imgs(records[species_recs_idx], species, rely_on_saved_urls=False,
                                                           image_column='image_url')
            rel_recs = records.loc[species_recs_idx & (records['image_url'].notnull()), :].copy()
            if not rel_recs.empty:
                records[species_recs_idx &
                        (records['image_url'].notnull())] = self.download_species_images(rel_recs,
                                                                                         image_column='image_url')

                species_and_recs = (
                    f"{self.dir_images}/{self.marker}/{species}", records.loc[species_recs_idx, :].copy())
                records[species_recs_idx] = self.mark_duplicates_by_image_for_dir(species_and_recs)

        records.loc[records['downloaded'].isnull(), 'downloaded'] = False
        records = records.loc[(records['image_url'].notnull()) &
                              ~(records['duplicate'].astype(bool)) &
                              (records['downloaded']), :].copy()

        len_after = len(records)
        removed_cnt = len_before - len_after
        self.log(f"Removed {removed_cnt} entries (duplicates/not downloaded) that could not be padded with images...")
        self.log(f"Keeping {len_after} entries...")

        records.to_csv(f"{self.dir_merged_recs}/{self.job_id}_{self.marker}.tsv", sep='\t', header=True, index=False)
        return records

    # extracts images + metadata from GBIF output
    def parse_gbif(self, gbif_out: dict, num_records: int = None, voucher: str = None) -> dict:
        urls = {}

        for record in gbif_out['results']:
            if num_records and len(urls) >= num_records:
                break
            if not voucher or re.search(rf'{re.escape(voucher)}[\W\D]|$', str(record)).group(0) != '':
                for media in record['media']:
                    if 'type' not in media:
                        continue
                    if media['type'] == 'StillImage':
                        try:
                            if re.search(r'.jpe?g$|$', media['identifier'], re.IGNORECASE).group(0) != '':
                                urls[media['identifier']] = {'license': media['license'],
                                                             'authors': media['rightsHolder'],
                                                             'gbif_observation_key': record['key']}
                            elif 'references' in media.keys() and re.search(r'.jpe?g+$|$', media['references'],
                                                                            re.IGNORECASE).group(0) != '':
                                urls[media['references']] = {'license': media['license'],
                                                             'authors': media['rightsHolder'],
                                                             'gbif_observation_key': record['key']}
                        except KeyError:
                            pass

        return urls
