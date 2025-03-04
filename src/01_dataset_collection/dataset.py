import sys

from Bio import AlignIO
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from glob import glob
import multiprocessing as mp
import numpy as np
import os
from pandarallel import pandarallel
import pandas as pd
from pathlib import Path
import re
import time
from tqdm import tqdm
from typing import Union

from barcode_filtering import BarcodeFiltering
from commons import Commons
from data_bold import Bold
from data_ncbi import NCBI
from data_padding import DataPadding
from server_prep import ServerPrep
from stats import Stats


class Dataset(Commons):
    def __init__(self, args):
        super().__init__(args.project_dir, args.job_id, args.num_processes)
        self.query = args.query
        self.query_type = args.query_type
        self.accessions = args.accessions
        self.fresh_images = args.fresh_images
        self.species_threshold = args.min_species
        self.threshold = args.threshold

        self.cluster_threshold: float = float()

        self.mkdirs()

        pandarallel.initialize(progress_bar=False, nb_workers=self.num_processes, verbose=0)

    # creates dataset_info.tsv if not exists, otherwise loads file
    def setup_dataset_info(self) -> pd.DataFrame:
        if os.path.isfile(f"{self.project_dir}/data/dataset_info.tsv"):
            overall_stats = pd.read_csv(f"{self.project_dir}/data/dataset_info.tsv", header=0, sep='\t')
            if self.job_id not in overall_stats['job_id'].values:
                overall_stats.loc[len(overall_stats), 'job_id'] = self.job_id
        else:
            overall_stats = pd.DataFrame(
                columns=['job_id', '#samples', '#species', '#genera', '#train', '#val', 'marker', 'cluster_thresh'])
            overall_stats.loc[len(overall_stats), 'job_id'] = self.job_id

        return overall_stats

    # updates data in dataset_info.tsv
    def update_dataset_info(self, ds: pd.DataFrame) -> None:
        overall_stats = self.setup_dataset_info()
        overall_stats.loc[overall_stats['job_id'] == self.job_id, :] = [self.job_id,
                                                                        len(ds),
                                                                        len(ds['species_name'].unique()),
                                                                        len(ds['genus_name'].unique()),
                                                                        len(ds[ds['dataset'] == 'val']),
                                                                        len(ds[ds['dataset'] == 'train']),
                                                                        self.marker,
                                                                        self.cluster_threshold
                                                                        ]
        overall_stats.to_csv(f"{self.project_dir}/data/dataset_info.tsv", header=True, index=False,
                             sep='\t')

    # checks if dataset was already compiled
    def dataset_compiled(self) -> bool:
        return len(glob(
            f"{self.project_dir}/data/{self.job_id}/records/merged/{self.job_id}_*.tsv")) > 0

    # wrapper function that starts dataset compilation process from BOLD
    def crawl_bold(self) -> list:
        self.log(f"========================\n{time.ctime()} - New run\n")

        bold = Bold(self.project_dir, self.job_id, self.query, self.query_type, self.num_processes)
        bold.mkdirs()
        bold.compile_dataset()
        families = bold.read_families_from_file()
        return families

    # NOTE: We provide family names in case BOLD container is provided instead of usual query
    def crawl_ncbi(self, families: list[str]) -> None:
        ncbi = NCBI(self.project_dir, self.job_id, self.num_processes, self.query_type, families)
        ncbi.compile_dataset(self.accessions)

    # wrapper function: generates stats on markers for choosing best marker
    def assess_markers(self, ds: pd.DataFrame) -> dict:
        marker_assessments = {}
        for marker in ds.loc[ds['in_dataset'], 'marker'].unique():
            m_ds = ds.loc[ds['marker'] == marker, :].copy()
            self.write_fastas(marker, m_ds)

            # cluster sub experiment check -> mark entries as outgroup where necessary
            bf = BarcodeFiltering(self.project_dir, self.job_id, self.num_processes, marker, self.threshold,
                                  self.species_threshold)
            bf.run_parallel(m_ds)

            # choose cluster per marker
            chosen_thresh, remaining_species, avg_records_per_species = bf.choose_cluster_thresh()
            if chosen_thresh is not None:
                marker_assessments[marker] = [chosen_thresh, remaining_species, avg_records_per_species]

        return marker_assessments

    # loads dataset file
    def load_dataset(self) -> pd.DataFrame:
        pattern = rf'{self.job_id}_[^_]+.tsv'
        ds_file = [f for f in glob(f"{self.project_dir}/data/{self.job_id}/records/merged/{self.job_id}_*.tsv")
                   if re.search(pattern, f)][0]

        ds = pd.read_csv(ds_file, sep='\t', header=0)
        return ds

    # wrapper function: generates DNA dataset by crawling data from BOLD and GENBANK
    # checks that sample threshold per species is met, removes duplicates, and chooses marker
    def compile_genetics(self) -> pd.DataFrame:
        if self.dataset_compiled():
            ds = self.load_dataset()
            self.marker = ds['marker'].values[0]
            cluster_df = pd.read_csv(f"{self.dir_base}/cluster_subexp/{self.marker}/cluster_info.tsv",
                                     sep='\t', header=0)
            self.cluster_threshold = cluster_df.loc[cluster_df['chosen'], 'id_thresh'].values[0]
        else:
            families = self.crawl_bold()
            self.crawl_ncbi(families)
            ds = self.concatenate_records()
            ds = self.apply_threshold(ds, save=True)  # apply threshold to get rid of rare markers/species
            ds = self.remove_ncbi_accession_duplicates(ds)
            ds = self.apply_threshold(ds, save=True)
            ds = self.choose_top_markers(ds)
            marker_assessments = self.assess_markers(ds)
            ds = self.choose_marker(marker_assessments, ds)
            self.cluster_threshold = marker_assessments[self.marker][0]

        return ds

    # wrapper function: generates image dataset by crawling from GBIF, checking that sample threshold is met,
    # and removing duplicates
    def compile_images(self, ds: pd.DataFrame) -> pd.DataFrame:
        if 'dataset' not in ds.columns:
            dpad = DataPadding(self.project_dir, self.job_id, self.num_processes, self.fresh_images, self.marker)
            ds = self.apply_threshold(ds, groupby=None)
            if self.fresh_images:  # remove images (bc usually preserved specimens)
                ds['image_urls'] = np.nan
                ds['duplicate'] = 'False'
            else:
                ds = self.remove_duplicate_urls(ds)
            ds = dpad.add_images(ds)
            ds = self.apply_threshold(ds, groupby=None, save=True)
            ds = self.download_images(ds)
            ds = self.remove_duplicate_images(ds)
            ds = self.split_records_into_single_url(ds)
            ds = dpad.quality_check_padding(ds)
            ds = self.apply_threshold(ds, groupby=None, save=True)
            return ds
        else:
            return ds

    # split into single-url entries
    def split_records_into_single_url(self, records: pd.DataFrame) -> pd.DataFrame:
        records = records.astype({'downloaded': 'str', 'duplicate': 'str'})

        if 'image_urls' in records.columns:
            records['image_urls'] = records['image_urls'].str.split('|')
            records['downloaded'] = records['downloaded'].str.split('|')
            records['duplicate'] = records['duplicate'].str.split('|')
            records = records.explode(['image_urls', 'downloaded', 'duplicate'])
            records.rename(columns={'image_urls': 'image_url'}, inplace=True)
            records.to_csv(f"{self.dir_merged_recs}/{self.job_id}_{self.marker}.tsv", header=True, index=False, sep='\t')
        return records

    def quality_check(self, ds: pd.DataFrame) -> pd.DataFrame:
        # first quality check
        if 'dataset' not in ds.columns:
            raise Warning("1st quality check needed. Use quality_filtering.ipynb")

        # padding + second quality check
        if 'train' not in ds['dataset'].values:
            dpad = DataPadding(self.project_dir, self.job_id, self.num_processes, self.fresh_images, self.marker)
            dpad.quality_check_padding(ds)
            raise Warning("2nd quality check needed. Use quality_filtering.ipynb")

        ds = self.apply_threshold(ds, save=True)

        # align barcodes and write aligned sequences to csv
        ServerPrep(self.project_dir, self.job_id, self.num_processes, self.marker, self.cluster_threshold).run(ds)
        # print statistics
        Stats(self.project_dir, self.job_id, self.num_processes, self.marker).result_statistics(ds)

        return ds

    # main function
    def compile(self) -> None:
        ds = self.compile_genetics()
        ds = self.compile_images(ds)
        ds = self.quality_check(ds)

        self.update_dataset_info(ds)

    # merges BOLD and GenBank datasets
    def concatenate_records(self) -> pd.DataFrame:
        concat_res_f = f"{self.dir_merged_recs}/{self.job_id}.tsv"

        if os.path.isfile(concat_res_f):
            self.log("Records already merged. Continuing...")
            records = pd.read_csv(concat_res_f, header=0, sep='\t')
        else:
            self.log("Merging records...")
            ncbi_recs = pd.read_csv(f"{self.dir_genbank}/GENBANK.tsv", header=0, sep='\t')
            bold_recs = pd.read_csv(f"{self.dir_bold}/BOLD.tsv", header=0, sep='\t')
            # bold record ID is only needed for backup + information is still available in BOLD.tsv
            bold_recs.drop(columns=['bold_record_id'], inplace=True)
            ncbi_recs['source'] = 'GENBANK'
            bold_recs['source'] = 'BOLD'

            records = pd.concat([bold_recs, ncbi_recs], ignore_index=True)

            # related_data is necessary for check later on
            records['related_data'] = records.apply(lambda x: True if x['image_cnt'] != 0 else False, axis=1)
            records.loc[:, 'marker'] = records.apply(
                lambda row: re.subn('(COI(?!-|I)|COX1|cox1|CO1)', 'COI-5P', row['marker'])[0], axis=1)
            records.loc[:, 'marker'] = records.apply(
                lambda row: re.subn('(rbcL(?!a|b))', 'rbcLa', row['marker'])[0], axis=1)
            records.loc[:, 'marker'] = records.apply(
                lambda row: re.subn('(EF-?1a)', 'EF-1a', row['marker'])[0], axis=1)
            records.loc[:, 'marker'] = records.apply(
                lambda row: re.subn(r'(?<=5\.8S)\sgene', '', row['marker'])[0], axis=1)
            records.loc[:, 'species_name'] = records.apply(
                lambda row: re.subn(r'\s(var\.|subsp\.).*', '', row['species_name'])[0], axis=1)
            records.loc[:, 'genus_name'] = records['species_name'].str.split(' ').str[0]
            records = records.loc[~records['species_name'].str.match(r'\w+\sagg.'), :]
            records['gbif_observation_key'] = None
            records.to_csv(concat_res_f, header=True, index=False, sep='\t')

        return records

    # chooses the 5 genetic markers/barcodes that are most prevalent in dataset and subsets dataframe
    def choose_top_markers(self, records: pd.DataFrame, top: int = 5) -> pd.DataFrame:
        if 'in_dataset' not in records.columns:
            markers = records.groupby('marker')['record_id'].count().nlargest(top).index.values
            records.loc[records['marker'].isin(markers), 'in_dataset'] = True
            records.loc[~records['marker'].isin(markers), 'in_dataset'] = False
        return records

    # chooses best marker according to barcode clustering/filtering and marker assessment (related to SNPs and gaps)
    def choose_marker(self, marker_assessments: dict, records: pd.DataFrame) -> pd.DataFrame:
        if len(glob(f"{self.dir_merged_recs}/{self.job_id}_*")) > 0:
            self.log("Best marker already chosen. Continuing...")
            m_records_f = glob(f"{self.dir_merged_recs}/{self.job_id}_*")[0]
            self.marker = re.search(rf'(?<={self.job_id}_)[^_]+', m_records_f).group(0)
            return pd.read_csv(f"{self.dir_merged_recs}/{self.job_id}_{self.marker}.tsv", sep='\t', header=0)

        self.log("Choosing best marker based on clustering results...")
        remaining_species_sorted = dict(sorted(marker_assessments.items(), key=lambda x: x[1][1]))
        avg_records_per_species_sorted = dict(sorted(marker_assessments.items(), key=lambda x: x[1][2]))

        remaining_species_sorted = list(np.array(list(remaining_species_sorted.values()))[:, 1])
        avg_records_per_species_sorted = list(np.array(list(avg_records_per_species_sorted.values()))[:, 2])

        marker_scores = {}
        for marker, marker_assessment in marker_assessments.items():
            chosen_thresh, remaining_species, avg_records_per_species = marker_assessment
            spec_cnt_idx = remaining_species_sorted.index(remaining_species)
            avg_species_idx = avg_records_per_species_sorted.index(avg_records_per_species)
            mean_idx = (spec_cnt_idx + avg_species_idx) / 2

            if mean_idx in marker_scores:
                marker_scores[mean_idx].append([marker, chosen_thresh])
            else:
                marker_scores[mean_idx] = [[marker, chosen_thresh]]

        best_marker = list(dict(sorted(marker_scores.items(), key=lambda x: x[0], reverse=True)).values())[0]

        if len(best_marker) > 1:
            best_marker = list(dict(sorted(best_marker, key=lambda x: x[1], reverse=True)).keys())[0]
        else:
            best_marker = best_marker[0][0]

        self.marker = best_marker
        id_thresh = marker_assessments[best_marker][0]

        aln_f = glob(f"{self.dir_base}/cluster_subexp/{best_marker}/id_{id_thresh}/*.aln")[0]
        msa = AlignIO.read(aln_f, 'fasta')

        record_ids = [record.id for record in msa]
        m_records = records.loc[records['record_id'].isin(record_ids), :]

        self.log(f"Best marker was {best_marker} with cluster threshold of {id_thresh}")
        self.log(f"Marker entries: {len(m_records)}")
        m_records.to_csv(f"{self.dir_merged_recs}/{self.job_id}_{best_marker}.tsv", header=True, index=False, sep='\t')
        return m_records

    # ensure that sample threshold is met and remove species with too few samples
    def apply_threshold(self, records: pd.DataFrame, groupby: Union[str, None] = 'marker',
                        save: bool = False) -> pd.DataFrame:
        if groupby:
            groups = ['species_name', groupby]
        else:
            groups = ['species_name']

        records.loc[:, 'group_cnts'] = records.groupby(groups)['species_name'].transform("count")
        len_before = len(records)
        records = records.loc[records['group_cnts'] >= self.threshold, :].copy()
        self.log(f"Removed {len_before - len(records)} entries with less than {self.threshold} entries...")
        self.log(f"Keeping {len(records)} entries with >={self.threshold} entries per species...")
        self.log(f"Keeping {len(records['species_name'].unique())} species with >={self.threshold} entries...")

        records.drop(columns=['group_cnts'], inplace=True)

        if save:
            suffix = f"_{self.marker}" if self.marker else ''
            records.to_csv(f"{self.dir_merged_recs}/{self.job_id}{suffix}.tsv", sep='\t', header=True, index=False)

        return records

    # deletes duplicate image files and information in dataframe
    def remove_duplicate_images(self, records: pd.DataFrame) -> pd.DataFrame:
        self.log("Removing duplicate images...")
        species_and_recs = [(f"{self.dir_images}/{self.marker}/{species}",
                             records.loc[records['species_name'] == species, :].copy())
                            for species in sorted(records['species_name'].unique())]

        records = records.astype({'duplicate': str})
        len_before = np.count_nonzero(records['duplicate'].str.split('|').explode().values == 'False')
        with mp.Pool(self.num_processes) as pool:
            new_records = list(tqdm(pool.imap(self.mark_duplicates_by_image_for_dir, species_and_recs),
                                    total=len(species_and_recs)))

        records = pd.concat(new_records, ignore_index=True)
        records = records.astype({'duplicate': str})
        len_after = np.count_nonzero(records['duplicate'].str.split('|').explode().values == 'False')
        self.log(f"Removed {len_before - len_after} records due to identical images...")
        records.to_csv(f"{self.dir_merged_recs}/{self.job_id}_{self.marker}.tsv", sep='\t', header=True, index=False)
        return records

    # iterate over dataframe and write barcodes to single-sequence fastas
    def write_fastas(self, marker: str, records: pd.DataFrame) -> None:
        if os.path.isfile(f"{self.dir_barcodes}/{marker}.fasta"):
            return

        Path(f"{self.dir_barcodes}/{marker}").mkdir(exist_ok=True, parents=True)
        records.parallel_apply(lambda row: self.write_fasta(row), axis=1)

    # writes single-sequence fasta from record
    def write_fasta(self, record: pd.Series) -> None:
        marker = record['marker']
        seq = record['sequence']
        record_id = record['record_id']
        species_name = record['species_name']
        sequence_file = f'{self.dir_barcodes}/{marker}/{record_id}.fasta'

        # remove existing alignment
        sequence = seq.replace('-', '')

        record = SeqRecord(
            Seq(sequence),
            id=record_id,
            description=f"{marker} {species_name.replace(' ', '_')}"
        )

        SeqIO.write(record, sequence_file, 'fasta')

    # ensure that no GenBank accession duplicates are in dataset
    def remove_ncbi_accession_duplicates(self, records: pd.DataFrame) -> pd.DataFrame:
        self.log("Removing accession duplicates...")

        # kick out duplicate accessions
        len_before = len(records)
        records_og = records.copy()

        records.drop_duplicates(subset='genbank_accession', inplace=True)
        new_records = []

        if len(records_og[records_og['image_urls'].notnull()]) > len(records[records['image_urls'].notnull()]):
            lost_imgs = records_og.loc[~records_og['image_urls'].isin(records['image_urls']), :].copy()
            for species in lost_imgs['species_name'].unique():
                lost_imgs_records = lost_imgs[lost_imgs['species_name'] == species].copy()
                records_wo_images = records[
                    (records['species_name'] == species) & (records['image_urls'].isnull())].copy()
                records_w_images = records[
                    (records['species_name'] == species) & (records['image_urls'].notnull())].copy()
                num_rows_to_pad = len(records_wo_images) - len(lost_imgs_records)

                new_records.append(self.add_records_w_new_imgs(num_rows_to_pad, lost_imgs_records,
                                                               records_wo_images, records_w_images))
        elif len(records_og) == len(records):
            return records

        records = pd.concat(new_records)
        self.log(f"Removed {len_before - len(records)} records based on duplicate accessions.")
        records.to_csv(f"{self.dir_merged_recs}/{self.job_id}.tsv", sep='\t', header=True, index=False)
        self.log(f"Keeping {len(records)} records.")
        return records

    # ensure that no duplicate URLs are in dataset
    def remove_duplicate_urls(self, records: pd.DataFrame) -> pd.DataFrame:
        if 'duplicate' not in records.columns:
            self.log("Removing URL duplicates...")

            # mark duplicate URLs
            records.sort_values(by='source', inplace=True)
            w_img_idx = records['image_urls'].notnull()
            rel_records = records.loc[w_img_idx, :].copy()
            total_records = pd.read_csv(f"{self.dir_merged_recs}/{self.job_id}.tsv", sep='\t', header=0)
            ambiguous_urls, dup_keep = self.collect_dup_urls(rel_records, total_records)

            records.loc[w_img_idx, 'duplicate'] = \
                records.loc[w_img_idx, :].apply(
                    lambda row: self.mark_duplicate(row, dup_keep, ambiguous_urls),
                    axis=1
                )
            records.to_csv(f"{self.dir_merged_recs}/{self.job_id}_{self.marker}.tsv", sep='\t', header=True,
                           index=False)
        else:
            self.log("URL duplicates already removed. Continuing...")
        return records

    # some records may have multiple URLs in the beginning. This method collects duplicate URLs from these multi-
    # URL entries
    def collect_dup_urls(self, records: pd.DataFrame, total_records: pd.DataFrame) -> tuple[list, dict]:
        dup_urls = {}
        records.loc[:, 'image_urls'] = records.loc[:, 'image_urls'].str.split('|')

        # collect urls that are connected to more than one species (check with complete dataset, not just subset)
        url_species_cnt = total_records.explode('image_urls')[['image_urls', 'species_name']].groupby('image_urls')[
            'species_name'].nunique()
        ambiguous_urls = url_species_cnt[url_species_cnt > 1].index.values

        for marker in records['marker'].unique():
            # split lists of urls within entry and expand to multiple rows
            url_df = records.loc[records['marker'] == marker, :].explode('image_urls').copy()
            # identify duplicates and drop all additional duplicates
            url_df = url_df[url_df.duplicated(subset=['image_urls'])].copy()
            url_df.drop_duplicates(subset=['image_urls'], inplace=True)
            dup_urls[marker] = dict(url_df[['image_urls', 'record_id']].values)

        return ambiguous_urls, dup_urls

    # update record information in dataframe if duplicate is found
    def mark_duplicate(self, record: pd.Series, dup_keep: dict, ambiguous_urls: list) -> str:
        if not isinstance(record['image_urls'], str):
            return np.nan

        marker = record['marker']
        img_urls = record['image_urls'].split('|')
        dup_urls = dup_keep[marker].keys()
        duplicate_info = ['False'] * len(img_urls)
        for i, img_url in enumerate(img_urls):
            if img_url in ambiguous_urls:
                duplicate_info[i] = 'True'
                continue

            if img_url in dup_urls:
                if not dup_keep[marker][img_url] == record['record_id']:
                    duplicate_info[i] = 'True'

        return '|'.join(duplicate_info)
