import numpy as np
import os
import pandas as pd
from pathlib import Path
import subprocess
from typing import Union

from commons import Commons


class Bold(Commons):
    def __init__(self, project_dir: Union[Path, str], job_id: str, query: str, query_type: str, num_processes: int):
        super().__init__(project_dir, job_id, num_processes)
        self.query_type = query_type
        self.query = query

    # main method - downloads and formats records
    def compile_dataset(self) -> None:
        if not os.path.isfile(f"{self.dir_bold}/BOLD.tsv"):
            self.download_records()
            self.format_records()
        else:
            self.log("BOLD records already fetched & processed. Continuing...")

    # download from BOLD API
    def download_records(self) -> None:
        bold_records_file = f'{self.dir_bold}/BOLD_original.tsv'
        if os.path.isfile(bold_records_file):
            return

        query = f'{self.query_type}={self.query.replace(" ", "%20")}'
        bold_url = f'http://v3.boldsystems.org/index.php/API_Public/combined?{query}&format=tsv'

        subprocess.run(['wget',
                        '-O',
                        bold_records_file,
                        bold_url],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE
                       )

    # wrapper for some reformatting that needs to happen for compatibility with NCBI and convenience
    def format_records(self) -> None:
        bold_records = pd.read_csv(f'{self.dir_bold}/BOLD_original.tsv', header=None, delimiter="\t",
                                   encoding="Windows-1252", encoding_errors='backslashreplace',
                                   dtype={11: str, 13: str, 15: str, 17: str, 19: str, 41: str})

        # it's interleaved, so we're gonna split it
        bold_records_reformatted = pd.DataFrame(bold_records.iloc[::2].reset_index(drop=True))
        bold_records_reformatted2 = pd.DataFrame(bold_records.iloc[1::2].reset_index(drop=True))

        # and join it back together
        bold_records_reformatted = bold_records_reformatted.join(bold_records_reformatted2, lsuffix='record',
                                                                 rsuffix='image')

        # formatting
        bold_records_reformatted = bold_records_reformatted.loc[:, bold_records_reformatted.iloc[0].notna()]
        bold_records_reformatted.columns = bold_records_reformatted.iloc[0]
        bold_records_reformatted = bold_records_reformatted[1:]

        bold_records_reformatted = self.remove_nas(bold_records_reformatted)
        bold_records_reformatted = self.count_images(bold_records_reformatted, with_dup=False)
        self.write_families_to_file(bold_records_reformatted)
        bold_records_reformatted = self.prepare_columns(bold_records_reformatted)

        bold_records_reformatted.to_csv(f'{self.dir_bold}/BOLD.tsv', sep='\t', index=False,
                                        header=True, encoding="UTF-8")

    # Nones are just empty strings at first, so we replace them with actual Nones
    def remove_nas(self, bold_records: pd.DataFrame) -> pd.DataFrame:
        bold_records['markercode'].replace(' ', np.nan, inplace=True)
        bold_records['species_name'].replace(' ', np.nan, inplace=True)
        bold_records.dropna(subset=['markercode', 'species_name'], inplace=True, how='any')

        bold_records = bold_records[~bold_records['species_name'].str.contains(' x ')]
        return bold_records

    # renaming and filtering of columns for later concatenation with NCBI records
    def prepare_columns(self, bold_records: pd.DataFrame) -> pd.DataFrame:
        bold_records.rename(columns={'nucleotides': 'sequence', 'markercode': 'marker', 'recordID': 'bold_record_id',
                                     'collectors': 'authors', 'copyright_licenses': 'license'},
                            inplace=True)
        bold_records.loc[:, 'taxonomy'] = bold_records.apply(
            lambda row: f"{row['phylum_name']}; {row['class_name']}; {row['order_name']}; {row['family_name']}", axis=1)
        bold_records = bold_records[["species_name", "marker", "sequence", "image_urls", "image_cnt", "authors",
                                     "license", "taxonomy", "genbank_accession", "bold_record_id"]]
        bold_records.insert(0, 'record_id', [f"BOLD{i}" for i in range(0, 0 + len(bold_records))])
        return bold_records

    # important in case BOLD container was provided instead of query family
    def write_families_to_file(self, bold_records: pd.DataFrame) -> None:
        with open(f"{self.dir_bold}/families.tsv", "w") as f:
            f.write('family_name\n')
            for family in bold_records['family_name'].unique():
                f.write(f"{family}\n")

    # important in case BOLD container was provided instead of query family
    def read_families_from_file(self) -> list:
        families = pd.read_csv(f"{self.dir_bold}/families.tsv", header=0)
        return families.loc[:, 'family_name'].unique()
