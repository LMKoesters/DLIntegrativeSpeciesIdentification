import ast
import glob
import http.client
from itertools import repeat
import math
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import requests
from retry import retry
from tqdm import tqdm
from typing import Union
import xml.parsers.expat
import xmltodict

from commons import Commons


class NCBI(Commons):
    def __init__(self, project_dir: Union[Path, str], job_id: str, num_processes: int,
                 query_type: str, query_families: list[str]):
        super().__init__(project_dir, job_id, num_processes)

        self.query_type = query_type
        self.query_families = query_families

    # main method - downloads and formats records
    def compile_dataset(self, accessions_tsv: Union[Path, str]) -> None:
        if not os.path.isfile(f"{self.dir_genbank}/GENBANK.tsv"):
            accessions = self.read_accessions(accessions_tsv)
            self.download_records(accessions)
            self.process_records()
            self.concatenate_batches()
        else:
            self.log("NCBI records already fetched & processed. Continuing...")

    # if file with list of GenBank accessions is provided, read from file to later get records
    def read_accessions(self, accessions_tsv: Union[Path, str]) -> list:
        if accessions_tsv:
            accessions = pd.read_csv(accessions_tsv, header=0, sep='\t')
            return accessions['accession'].values

        return [None]

    # downloads records from NCBI API (takes some time)
    def download_records(self, accessions: list) -> None:
        if len(glob.glob(f"{self.dir_genbank}/GENBANK_*.tsv")) != 0 and not os.path.isfile(
                f"{self.dir_genbank}/GENBANK_IDs_0.txt"):
            self.log("GENBANK records already fetched. Continuing...")
            return

        self.log('Fetching records...')
        if accessions[0]:
            total_records = len(accessions)
        else:
            total_records = self.ncbi_esearch(return_count=True)

        batches = math.ceil(total_records / 10_000)

        with mp.Pool(self.num_processes) as pool:
            list(tqdm(pool.imap(self.download_batch, zip(range(0, batches), repeat(accessions))), total=batches))

        self.remove_temporary_files()

    # included other functionality once upon time, now just adds markers to records
    def process_records(self) -> None:
        self.log('Processing NCBI records...')
        batch_tsvs = glob.glob(f'{self.dir_genbank}/GENBANK_records*.tsv')

        with mp.Pool(self.num_processes) as pool:
            list(tqdm(pool.imap(self.add_marker, batch_tsvs), total=len(batch_tsvs)))

    # we downloaded records in batches, so we need to concatenate them
    def concatenate_batches(self) -> None:
        self.log('Concatenating NCBI records...')
        records = glob.glob(f'{self.dir_genbank}/GENBANK_records*.tsv')
        records = pd.concat(
            [pd.read_csv(ncbi_batch, header=0, delimiter='\t') for ncbi_batch in records],
            ignore_index=True)

        # compatibility
        records['image_cnt'] = 0
        records['image_urls'] = None

        records = records.astype({"image_cnt": int})
        records = records[["species_name", "marker", "sequence", "image_urls", "image_cnt",
                           "GBSeq_taxonomy", "GBSeq_primary-accession"]].copy()
        records.rename(columns={'GBSeq_taxonomy': 'taxonomy'}, inplace=True)
        records.rename(columns={'GBSeq_primary-accession': 'genbank_accession'}, inplace=True)
        records.dropna(subset=['marker'], inplace=True)
        records.insert(0, 'record_id', [f"GENBANK{i}" for i in range(0, 0 + len(records))])
        records.to_csv(f'{self.dir_genbank}/GENBANK.tsv',
                       sep='\t',
                       index=False,
                       header=True)

    # download batch of records from NCBI API
    def download_batch(self, args) -> None:
        batch_num = args[0]
        accessions = args[1]
        lower_limit = batch_num * 10_000
        upper_limit = min(lower_limit + 10_000, len(accessions))
        offset = lower_limit if None in accessions else 0

        # esearch part
        if not accessions[0]:
            self.ncbi_esearch(offset=offset, batch_num=batch_num)
        else:
            for i, accession in enumerate(accessions[lower_limit:upper_limit]):
                self.ncbi_esearch(offset=offset, batch_num=batch_num, accession=accession, accession_i=i)
            if not os.path.isfile(f"{self.dir_genbank}/GENBANK_IDs_{batch_num}.txt"):
                with open(f"{self.dir_genbank}/GENBANK_IDs_{batch_num}.txt", "w+") as f:
                    for accession_id_txt in glob.glob(f"{self.dir_genbank}/GENBANK_IDs_{batch_num}_*.txt"):
                        with open(accession_id_txt) as accession_id:
                            f.write(accession_id.read())

                        os.remove(accession_id_txt)

        self.ncbi_efetch(args[0])

        if not self.xml_to_tsv(args[0]):
            self.download_batch(args)

    @retry(KeyError, delay=1, tries=100)
    def ncbi_esearch(self, return_count: bool = False, offset: int = None,
                     batch_num: int = None, accession: str = None, accession_i: int = None) -> Union[int, None]:
        api_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'

        query = f"({'+OR+'.join(self.query_families).replace(' ', '+')})"

        if self.query_type == 'taxon':
            term = f'(misc+RNA[Feature+key]+OR+gene+[Feature+key])+AND+{query}+AND+biomol+genomic[Properties]'
        else:
            term = accession

        if return_count:
            response = requests.get(f'{api_url}?db=nucleotide&term={term}&rettype=count')
            return int(xmltodict.parse(response.content)['eSearchResult']['Count'])
        elif accession_i is not None:
            ids_txt = f"{self.dir_genbank}/GENBANK_IDs_{batch_num}_{accession_i}.txt"
        else:
            ids_txt = f"{self.dir_genbank}/GENBANK_IDs_{batch_num}.txt"

        if os.path.isfile(ids_txt) or os.path.isfile(f"{self.dir_genbank}/GENBANK_records_{batch_num}.xml"):
            return

        response = requests.get(f'{api_url}?db=nucleotide&term={term}&retmode=json&retmax=10000&retstart={offset}')
        id_list = response.json()['esearchresult']['idlist']
        with open(ids_txt, 'w+') as complete_f:
            [complete_f.write(f'{idx}\n') for idx in id_list]

    @retry(xml.parsers.expat.ExpatError, delay=1)
    def ncbi_efetch(self, batch_num: int) -> None:
        if os.path.isfile(f'{self.dir_genbank}/GENBANK_records_{batch_num}.xml'):
            return

        try:
            ids_batch = np.loadtxt(f'{self.dir_genbank}/GENBANK_IDs_{batch_num}.txt',
                                   dtype=str)
            try:
                ids_str = ",".join(ids_batch)
            except TypeError:
                ids_str = str(ids_batch)

            data = {'db': 'nucleotide',
                    'id': ids_str,
                    'retmode': 'xml',
                    'email': 'lara.koesters@tu-ilmenau.de'}

            # process_idx = mp.current_process()._identity[0]
            process_idx = 1

            with requests.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                               stream=True, data=data, timeout=10) as r:
                with open(f"{self.dir_genbank}/GENBANK_records_{batch_num}.xml", 'w') as fd, tqdm(
                        desc=f'GENBANK_{batch_num}',
                        unit='B',
                        unit_scale=True,
                        position=process_idx) as bar:
                    for records_xml in r.iter_content(chunk_size=10_000, decode_unicode=True):
                        fd.write(records_xml)
                        bar.update(10_000)
        except requests.exceptions.ConnectionError:
            print(f"Failed to download {batch_num} records. Retrying...")
            os.remove(f"{self.dir_genbank}/GENBANK_records_{batch_num}.xml")
            raise xml.parsers.expat.ExpatError

    # turns GenBank output XML files into TSVs
    def xml_to_tsv(self, batch_num: int) -> bool:
        efetch_tsv = f"{self.dir_genbank}/GENBANK_records_{batch_num}.tsv"
        if os.path.isfile(efetch_tsv):
            return True

        try:
            records_dict = xmltodict.parse(Path(f"{self.dir_genbank}/GENBANK_records_{batch_num}.xml").read_text())
        except (xml.parsers.expat.ExpatError, TypeError):
            os.remove(f"{self.dir_genbank}/GENBANK_records_{batch_num}.xml")
            return False

        records = pd.DataFrame.from_records(records_dict['GBSet']['GBSeq'])
        records = records[~records['GBSeq_organism'].str.contains(' x ')]
        records.rename(columns={'GBSeq_organism': 'species_name'}, inplace=True)
        records.rename(columns={'GBSeq_sequence': 'sequence'}, inplace=True)
        records.to_csv(efetch_tsv, sep='\t', header=True, index=False)
        return True

    # removes files from GenBank download that are no longer needed
    def remove_temporary_files(self) -> None:
        ids = glob.glob(f"{self.dir_genbank}/GENBANK_IDs*")
        xmls = glob.glob(f"{self.dir_genbank}/GENBANK_records*.xml")

        for id_f in ids:
            os.remove(id_f)
        for xml in xmls:
            os.remove(xml)

    # adds marker to rows in GenBank batch dataframe
    def add_marker(self, batch_tsv: Union[Path, str]) -> None:
        records = pd.read_csv(batch_tsv, sep='\t', header=0)
        if len(records) == 0 or 'marker' in records.columns:
            return

        records = records[~records['species_name'].str.contains(' x ')].copy()
        records = records[~records['GBSeq_definition'].str.contains('complete genome')].copy()
        markers = records.apply(lambda row: self.retrieve_record_marker(row), axis=1)

        if len(markers) == 0:  # no gene entries found
            records.loc[:, 'marker'] = np.nan
        else:
            records.loc[:, 'marker'] = markers

        records.dropna(subset=['marker', 'species_name'], inplace=True, how='any')
        records.to_csv(batch_tsv, sep='\t', index=False, header=True)

    # finds marker for record row by parsing feature table cell
    def retrieve_record_marker(self, row: pd.Series) -> Union[float, str]:
        feature_table = ast.literal_eval(row['GBSeq_feature-table'])
        feature_table = pd.DataFrame.from_dict(feature_table['GBFeature'])

        markers = feature_table.apply(lambda x:
                                      self.marker_from_features(x),
                                      axis=1)

        # no gene entries found
        if type(markers) == float:
            return np.nan

        feature_table['marker_GBQualifier_value'] = markers
        feature_table.dropna(subset='marker_GBQualifier_value', inplace=True)

        marker = '|'.join(feature_table['marker_GBQualifier_value'].unique())

        if not marker:
            marker = re.search(r'(Ranunculus\s\w+\s)(.*;\s)?(.*)(?=.*,\scomplete)|$', row['GBSeq_definition']).group(3)

        if marker:
            split_pattern = re.compile(r',?\sand\s|,\s')
            marker = '|'.join(set(re.split(split_pattern, marker)))
            marker = marker.replace('internal transcribed spacer ', 'ITS')
            marker = marker.replace(' ribosomal RNA gene', '')
            return marker
        else:
            return np.nan

    # marker from feature table inside GenBank dataframe cell
    def marker_from_features(self, row: pd.Series) -> Union[float, str]:
        try:
            feature_quals = row['GBFeature_quals']
            gb_qualifier = feature_quals['GBQualifier']
            if isinstance(gb_qualifier, list):
                gb_qualifier_df = pd.DataFrame.from_records(feature_quals['GBQualifier'])
            elif isinstance(gb_qualifier, dict):
                gb_qualifier_df = pd.DataFrame.from_records([feature_quals['GBQualifier']])
            else:
                return np.nan

            markers = gb_qualifier_df.loc[gb_qualifier_df['GBQualifier_name'] == 'gene', 'GBQualifier_value'].values
            return markers[0] if len(markers) > 0 else np.nan
        except (TypeError, KeyError):
            return np.nan

