from Bio import AlignIO
import glob
import sys

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import subprocess
from tqdm import tqdm

from commons import Commons


class BarcodeFiltering(Commons):
    SEQ_LENGTH = {
        # Lycaenidae
        'EF-1a': 1000,  # not enough samples either way
        # Lycaenidae + Coccinellidae
        'COI-5P': 660,  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6231476/
        'ND5': 1000,  # not enough samples either way
        'COII': 1000,  # not enough samples either way
        'COI-3P': 809,  # https://www.biorxiv.org/content/10.1101/2020.11.23.394510v1
        # Coccinellidae
        'H3': 1000,  # not enough samples either way
        # Asteraceae
        'ITS2': 400,  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7162488/
        'rpl32': 1000,  # not enough samples either way
        # Asteraceae + Poaceae
        'rbcLa': 670,  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9390329/
        'matK': 800,  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8427591/
        'rps16': 880,  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5701390/
        # Poaceae
        'BADH2': 1000,  # not enough samples either way
        'ndhF': 1000,  # not enough samples either way
    }

    def __init__(self, project_dir, job_id, num_processes, marker, threshold, species_threshold):
        super().__init__(project_dir, job_id, num_processes)

        self.species_threshold = species_threshold
        self.threshold = threshold
        self.marker = marker
        self.cluster_dir = f'{self.dir_base}/cluster_subexp/{marker}'
        self.in_fsa = self.cat_fastas()
        self.max_seq_length = self.SEQ_LENGTH[self.marker] if self.marker in self.SEQ_LENGTH else 1000
        self.min_seq_length = int(self.max_seq_length * 0.4)
        self.cluster_steps = [round(id_thresh, 2) for id_thresh in np.arange(0.5, 1., 0.01)]

        self.mkdirs()

    def mkdirs(self):
        for d in [self.cluster_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

    # concatenate single-sequence fastas
    def cat_fastas(self):
        marker_fsa = f"{self.dir_barcodes}/{self.marker}.fasta"
        if not os.path.isfile(marker_fsa):
            os.system(f"find '{self.dir_barcodes}/{self.marker}' " + "-type f -exec cat {} + >" + f'"{marker_fsa}"')
        return marker_fsa

    # wrapper function for parallelization
    def run_parallel(self, records):
        if len(glob.glob(f"{self.dir_merged_recs}/{self.job_id}_*.tsv")) > 0 or \
                os.path.isfile(f"{self.cluster_dir}/cluster_info.png"):
            return

        self.cat_fastas()

        with mp.Pool(self.num_processes) as pool:
            list(tqdm(pool.imap(self.run, self.cluster_steps), total=len(self.cluster_steps)))

        self.collect_main_cluster_info(records)

    # run barcode clustering
    def run(self, id_thresh):
        id_dir = f"{self.cluster_dir}/id_{id_thresh}"
        Path(id_dir).mkdir(parents=True, exist_ok=True)

        self.vsearch(id_thresh, id_dir)
        largest_cluster = self.find_largest_cluster(id_dir)
        if largest_cluster:
            cluster_id = re.search(r'(?<=clusterfasta)\d+', largest_cluster).group(0)
            aln = self.build_alignment(id_dir, largest_cluster, cluster_id)
            aln_length = self.aln_length(aln)
            vcf = self.snp_sites(id_dir, aln, cluster_id)
            self.vcf_tools(id_dir, vcf, cluster_id, aln_length)

    def vsearch(self, id_thresh, id_dir):
        if os.path.isfile(f"{id_dir}/{self.marker}_cluster_centroids.fasta"):
            return

        p = subprocess.Popen(["vsearch",
                              "--threads", f"{1}",
                              "--minseqlength", f"{self.min_seq_length}",
                              "--maxseqlength", f"{self.max_seq_length}",
                              "--id", f"{id_thresh}",
                              "--cluster_size", f"{self.in_fsa}",
                              "--clusters", f"{id_dir}/{self.marker}_clusterfasta",
                              "--centroids", f"{id_dir}/{self.marker}_cluster_centroids.fasta",
                              "--consout", f"{id_dir}/{self.marker}_cluster_consensus.fasta"
                              ],
                             cwd=os.getcwd(),
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        _output, _error = p.communicate()

    # count sequences in clusters
    def find_largest_cluster(self, id_dir):
        cluster_sizes = {}

        for f in glob.glob(f"{id_dir}/{self.marker}_clusterfasta*"):
            p = subprocess.Popen(f"grep -c '>' '{f}'",
                                 shell=True,
                                 cwd=os.getcwd(),
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE
                                 )

            seq_cnt, error = p.communicate()
            try:
                seq_cnt = int(seq_cnt.decode().strip())
                cluster_sizes[f] = seq_cnt
            except:
                print(self.marker)
                print(id_dir)
                print(error)
                sys.exit()

        try:
            return max(cluster_sizes, key=lambda k: cluster_sizes[k])
        except ValueError:
            return None

    def build_alignment(self, id_dir, largest_cluster, cluster_id):
        aln = f"{id_dir}/{self.marker}_cluster{cluster_id}.aln"

        if not os.path.isfile(aln):
            os.system(f"mafft --auto --quiet {largest_cluster} > {aln}")

        return aln

    def snp_sites(self, id_dir, aln, cluster_id):
        vcf = f"{id_dir}/{self.marker}_cluster{cluster_id}.vcf"

        if not os.path.isfile(vcf):
            p = subprocess.Popen(["snp-sites",
                                  "-v",
                                  "-o", vcf,
                                  aln
                                  ],
                                 cwd=os.getcwd(),
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

            _output, _error = p.communicate()

        return vcf

    def vcf_tools(self, id_dir, vcf, cluster_id, aln_length):
        snps_f = f"{id_dir}/{self.marker}_cluster{cluster_id}"

        if not os.path.isfile(f"{snps_f}.snpden"):
            p = subprocess.Popen(["vcftools",
                                  "--vcf", vcf,
                                  "--SNPdensity", str(aln_length),
                                  "--out", snps_f
                                  ],
                                 cwd=os.getcwd(),
                                 stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)

            _output, _error = p.communicate()

    def aln_length(self, aln_f):
        try:
            msa = AlignIO.read(aln_f, 'fasta')
        except:
            print(aln_f)
            sys.exit()
        return len(msa[0].seq)

    # gathers information on clusters (i.e. 1 cluster/alignment file per identity threshold) for
    # marker assessment + choosing the best marker
    def collect_main_cluster_info(self, records):
        if not os.path.isfile(f"{self.cluster_dir}/cluster_info.tsv"):
            cols = ['id_thresh', 'gap_count', 'gap_openings', 'gap_extensions', 'gap_score', 'rel_gap_score',
                    'avg_gap_length', 'cluster_size', 'marker_recs', 'species_cnt', 'remaining_recs',
                    'remaining_species', 'avg_records_per_species', 'aln_length']
            cluster_info = {col: [] for col in cols}
            snpdens = []

            for id_thresh in self.cluster_steps:
                try:
                    aln_f = glob.glob(f"{self.cluster_dir}/id_{id_thresh}/*.aln")[0]
                except IndexError:
                    continue

                msa = AlignIO.read(aln_f, 'fasta')

                # snpdens
                try:
                    snpden_f = glob.glob(f"{self.cluster_dir}/id_{id_thresh}/*.snpden")[0]
                    id_thresh = float(re.search(r'(?<=id_)[\d.]+', snpden_f).group(0))
                    id_df = pd.read_csv(snpden_f, sep='\t', header=0)
                    id_df['id_thresh'] = id_thresh
                    snpdens.append(id_df)
                except IndexError:
                    pass

                # species count
                total_records = records.copy()
                total_records.loc[:, 'group_cnts'] = total_records.groupby('species_name')['species_name'].transform(
                    "count")
                total_records = total_records.loc[total_records['group_cnts'] >= self.threshold, :]
                marker_recs = len(total_records)
                species_cnt = len(total_records['species_name'].unique())

                remaining_ids = [record.id for record in msa]
                remaining_records = records.loc[records['record_id'].isin(remaining_ids), :].copy()
                remaining_records.loc[:, 'group_cnts'] = remaining_records.groupby('species_name')[
                    'species_name'].transform("count")
                remaining_records = remaining_records.loc[remaining_records['group_cnts'] >= self.threshold, :]
                remaining_recs = len(remaining_records)
                remaining_species = len(remaining_records['species_name'].unique())

                try:
                    avg_rec_per_species = round(remaining_recs / remaining_species, 2)
                except ZeroDivisionError:
                    avg_rec_per_species = 0

                # gap score
                gap_count = sum([record.seq.count('-') for record in msa])
                gap_openings = sum([len(re.findall(r'-+', str(record.seq))) for record in msa])
                gap_extensions = gap_count - gap_openings
                gap_score = (3 * gap_openings + gap_extensions) / len(msa)
                rel_score = round(gap_score / float(len(msa[0].seq)), 2)

                # gap length
                gaps = [re.findall(r'-+', str(record.seq)) for record in msa]
                avg_gap_length = sum([len(gap) for gap in gaps]) / len(gaps)

                # misc info
                cluster_size = len(msa)
                aln_length = self.aln_length(glob.glob(f"{self.cluster_dir}/id_{id_thresh}/*.aln")[0])

                [cluster_info[k].append(v) for k, v in zip(cols, [id_thresh, gap_count, gap_openings, gap_extensions,
                                                                  gap_score, rel_score, avg_gap_length, cluster_size,
                                                                  marker_recs, species_cnt, remaining_recs,
                                                                  remaining_species, avg_rec_per_species, aln_length])]

            try:
                snpdens = pd.concat(snpdens, ignore_index=True)
            except ValueError:
                return

            df = pd.DataFrame.from_dict(cluster_info)
            df = df.merge(snpdens, on='id_thresh', how='left')

            df.loc[df['SNP_COUNT'].isnull(), 'SNP_COUNT'] = 0
            df['rel_snp_count'] = round(df['SNP_COUNT'] / df['aln_length'], 4)

            df['rel_remaining_species'] = df['remaining_species'] / df['species_cnt']
            df['rel_remaining_recs'] = df['remaining_recs'] / df['marker_recs']

            df.sort_values(by='id_thresh', inplace=True)
            df.to_csv(f"{self.cluster_dir}/cluster_info.tsv", sep='\t', header=True, index=False)

    # visualize cluster information
    def plot_cluster_info(self, id_thresh, max_thresh, suffix=''):
        cluster_df = pd.read_csv(f"{self.cluster_dir}/cluster_info.tsv", sep='\t', header=0)
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex='all', figsize=(10, 13))
        upper_lim = min(max(
            max(cluster_df['rel_snp_count']),
            max(cluster_df['rel_gap_score'])) + 0.1, 1.0)

        # Plot the fitted curve
        min_gap = min(cluster_df[cluster_df['id_thresh'] <= max_thresh]['rel_gap_score'])
        ax1.plot(cluster_df['id_thresh'], cluster_df['rel_gap_score'], color='#00b3b3', label='gap score')
        ax1.hlines(min_gap, xmin=min(cluster_df['id_thresh']), xmax=max(cluster_df['id_thresh']), linestyles='--',
                   lw=1, colors='#00b3b3')
        ax1.vlines(id_thresh, ymin=0, ymax=upper_lim, linestyles='--', lw=1, colors='#00b3b3')
        ax1.set_ylabel('GAP SCORE')

        max_snp = max(cluster_df[cluster_df['id_thresh'] <= max_thresh]['rel_snp_count'])
        ax2.plot(cluster_df['id_thresh'], cluster_df['rel_snp_count'], color='purple', label='rel snp count')
        ax2.hlines(max_snp, xmin=min(cluster_df['id_thresh']), xmax=max(cluster_df['id_thresh']), linestyles='--',
                   lw=1, colors='purple')
        ax2.vlines(id_thresh, ymin=0, ymax=upper_lim, linestyles='--', lw=1, colors='purple')
        ax2.set_ylabel('REL SNP COUNT')

        ax3.bar(cluster_df['id_thresh'], cluster_df['remaining_species'], label='remaining species',
                fill=False, width=0.01)
        ax3.vlines(id_thresh, ymin=0, ymax=max(cluster_df['remaining_species']), linestyles='--', lw=1, colors='black')
        ax3.set_ylabel('REMAINING SPECIES')

        ax4.bar(cluster_df['id_thresh'], cluster_df['remaining_species'] / cluster_df['species_cnt'],
                label='rel remaining species', fill='blue', alpha=0.5, width=0.01)
        ax4.vlines(id_thresh, ymin=0, ymax=upper_lim, linestyles='--', lw=1, colors='blue', alpha=0.5)
        ax4.set_ylabel('REL REMAINING SPECIES')

        ax5.bar(cluster_df['id_thresh'], cluster_df['remaining_recs'] / cluster_df['marker_recs'],
                label='rel remaining records', fill='blue', alpha=0.5, width=0.01)
        ax5.vlines(id_thresh, ymin=0, ymax=upper_lim, linestyles='--', lw=1, colors='blue', alpha=0.5)
        ax5.set_ylabel('REL REMAINING RECORDS')

        ax1.set_ylim(0, upper_lim)
        ax2.set_ylim(0, upper_lim)

        plt.rc('axes', labelsize=5)
        ax5.set_xlabel('CLUSTER THRESHOLD')

        plt.tight_layout()

        plt.savefig(f"{self.cluster_dir}/cluster_info{suffix}.png", dpi=299)
        plt.close(fig)

    # choose the identity threshold with the best features
    def choose_cluster_thresh(self):
        try:
            cluster_df = pd.read_csv(f"{self.cluster_dir}/cluster_info.tsv", sep='\t', header=0)
        except FileNotFoundError:
            return None, 0, 0
        cluster_df = cluster_df.loc[cluster_df['remaining_species'] >= self.species_threshold, :].copy()
        if cluster_df.empty:
            return None, 0, 0

        max_snp = max(cluster_df['rel_snp_count'])
        min_gap = min(cluster_df['rel_gap_score'])

        id_scores = {}
        for id_thresh in cluster_df['id_thresh'].unique():
            rel_snp, rel_gap, rel_species = \
                cluster_df.loc[cluster_df['id_thresh'] == id_thresh, ['rel_snp_count', 'rel_gap_score',
                                                                      'rel_remaining_species']].values[0]
            snp_diff = max_snp - rel_snp
            gap_diff = 2 * (rel_gap - min_gap)
            mean_diff = (snp_diff + gap_diff) / 2
            id_scores[id_thresh] = mean_diff

        id_scores = sorted(id_scores.items(), key=lambda x: x[1])
        chosen_id = id_scores[0][0]
        rel_cluster_info = cluster_df.loc[cluster_df['id_thresh'] == chosen_id, :]

        self.plot_cluster_info(chosen_id, max(cluster_df['id_thresh']))

        cluster_df['chosen'] = False
        cluster_df.loc[cluster_df['id_thresh'] == chosen_id, 'chosen'] = True
        cluster_df.to_csv(f"{self.cluster_dir}/cluster_info.tsv", sep='\t', header=True, index=False)

        return chosen_id, rel_cluster_info['remaining_species'].values[0], \
               rel_cluster_info['avg_records_per_species'].values[0]
