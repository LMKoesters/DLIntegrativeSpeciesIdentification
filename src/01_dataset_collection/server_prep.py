from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
import numpy as np
import os.path
import pandas as pd
from pathlib import Path
from PIL import Image

from commons import Commons


class ServerPrep(Commons):
    one_hot_encoding = {'A': [1., 0., 0., 0.],
                        'T': [0., 1., 0., 0.],
                        'C': [0., 0., 1., 0.],
                        'G': [0., 0., 0., 1.],
                        'Y': [0., 0.5, 0.5, 0.],
                        'R': [0.5, 0., 0., 0.5],
                        'W': [0.5, 0.5, 0., 0.],
                        'S': [0., 0., 0.5, 0.5],
                        'K': [0., 0.5, 0., 0.5],
                        'M': [0.5, 0., 0.5, 0.],
                        'D': [.33, .33, 0., .33],
                        'V': [.33, 0., .33, .33],
                        'H': [.33, .33, .33, 0.],
                        'B': [0., .33, .33, .33],
                        '-': [0., 0., 0., 0.],
                        'N': [.25, .25, .25, .25]}

    def __init__(self, project_dir, job_id, num_processes, marker, cluster_thresh):
        super().__init__(project_dir, job_id, num_processes)

        self.marker = marker
        self.cluster_thresh = cluster_thresh

        self.dir_server_prep = f"{self.dir_base}/server_prep"
        self.dir_server_img = f"{self.dir_server}/images"
        self.dir_server_bar = f"{self.dir_server}/barcodes"

        self.mkdirs()

    def mkdirs(self):
        super().mkdirs()

        for d in [self.dir_server_prep, self.dir_server_img, self.dir_server_bar]:
            Path(d).mkdir(parents=True, exist_ok=True)

    # main method - encodes barcodes and sorts both barcodes and images into needed folder structure
    def run(self, records):
        self.log('Preparing server directory...')
        records = self.prepare_barcodes(records)
        self.encode_barcodes(records)
        self.sort_images(records)

    # adds aligned and SNP-reduced barcodes
    def prepare_barcodes(self, records):
        if 'unaligned_barcode' not in records.columns:
            records.apply(lambda row: self.add_seq_msa(row), axis=1)
            self.align_barcodes()
            records = self.add_barcode_cols(records)
            records.to_csv(f"{self.dir_merged_recs}/{self.job_id}_{self.marker}.tsv", sep='\t',
                           header=True, index=False)

        return records

    # writes sequence to multi-sequence-fasta
    def add_seq_msa(self, record):
        seq = record['sequence']
        record_id = record['record_id']
        species_name = record['species_name']
        sequence_file = f'{self.dir_server_prep}/{self.marker}.fasta'

        # remove existing alignment
        sequence = seq.replace('-', '')

        record = SeqRecord(
            Seq(sequence),
            id=record_id,
            description=f"{self.marker} {species_name.replace(' ', '_')}"
        )

        with open(sequence_file, 'a') as f:
            SeqIO.write(record, f, 'fasta')

    # align barcodes with MAFFT and calculate SNPs
    def align_barcodes(self):
        in_fa = f"{self.dir_server_prep}/{self.marker}.fasta"
        out_aln = f"{self.dir_server_prep}/{self.marker}.aln"

        os.system(
            f'mafft --quiet {in_fa} > {out_aln}')
        msa = AlignIO.read(out_aln, 'fasta')
        aln_length = len(msa[0].seq)

        vcf = f"{self.dir_server_prep}/{self.marker}.vcf"
        os.system(f'snp-sites -v -o {vcf} {out_aln}')

        snps_f = f"{self.dir_server_prep}/{self.marker}"
        os.system(f'vcftools --vcf {vcf} --SNPdensity {aln_length} --out {snps_f}')

    # fetches from aligned barcodes file + adds raw barcodes column
    def add_barcode_cols(self, records):
        aln_f = f"{self.dir_server_prep}/{self.marker}.aln"
        msa = AlignIO.read(aln_f, 'fasta')

        # 1-based position!
        vcf = pd.read_csv(f"{self.dir_server_prep}/{self.marker}.vcf", header=3, sep='\t')
        vcf_positions = vcf['POS'].values

        for record in msa:
            record_id = record.id
            records.loc[records['record_id'] == record_id, 'aligned_barcode'] = str(record.seq)
            records.loc[records['record_id'] == record_id,
                        'aligned_barcode_snp'] = ''.join([ch if i + 1 in vcf_positions else '' for i, ch in
                                                          enumerate(str(record.seq))])

        records.loc[:, 'padded_barcode'] = records.apply(
            lambda x: x['sequence'].replace('-', '') + (
                        '-' * (len(x['aligned_barcode']) - len(x['sequence'].replace('-', '')))), axis=1)
        records.loc[:, 'unaligned_barcode'] = records.apply(lambda x: x['sequence'].replace('-', ''), axis=1)

        return records

    def encode_barcodes(self, records):
        for e in ['aligned_barcode', 'padded_barcode', 'aligned_barcode_snp', 'unaligned_barcode']:
            s = 'one_hot'
            m = self.one_hot_encoding
            d = f"{self.dir_server_bar}/{e}/{s}_bar"
            Path(d).mkdir(parents=True, exist_ok=True)

            records.apply(lambda row: self.encode_barcode(row, m, d, e), axis=1)

    def encode_barcode(self, row, m, d, e):
        species_d = f"{d}/{row['dataset']}/{row['species_name'].replace(' ', '_')}"
        Path(species_d).mkdir(parents=True, exist_ok=True)

        t = [m[i] for i in str(row[e]).upper()]
        np.save(f"{species_d}/{row['record_id']}.npy", np.array(t))

    def sort_images(self, records):
        for ds in ['train', 'val']:
            d = f"{self.dir_server}/images/{ds}"
            Path(d).mkdir(parents=True, exist_ok=True)

        records.loc[:, 'image_path'] = records.apply(
            lambda row: self.copy_image_to_server_folder(row), axis=1)
        records.to_csv(f"{self.dir_merged_recs}/{self.job_id}_{self.marker}.tsv", sep='\t',
                       header=True, index=False)

    # copies image from intermediate folder to final server folder
    def copy_image_to_server_folder(self, record):
        try:
            old_f_path = f"{self.dir_images}/{record['marker']}/{record['species_name']}/{record['record_id']}"
            img = Image.open(old_f_path)
        except FileNotFoundError:
            old_f_path = record['image_path']
            img = Image.open(old_f_path)

        species_dir = f"{self.dir_server_img}/{record['dataset']}/{record['species_name'].replace(' ', '_')}"
        Path(species_dir).mkdir(exist_ok=True, parents=True)
        new_f_path = f"{species_dir}/{record['record_id']}.tiff"

        img.save(new_f_path, compression=None)
        return new_f_path
