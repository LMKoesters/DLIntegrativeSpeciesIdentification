from collections import defaultdict
import os
from itertools import repeat
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import subprocess
import sys

from commons import Commons


class Stats(Commons):
    def __init__(self, project_dir, job_id, num_processes, marker):
        super().__init__(project_dir, job_id, num_processes)
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.marker = marker

        self.dir_stats = os.path.join(self.dir_base, 'stats')
        self.dir_results = os.path.join(self.project_dir, 'results')
        self.dir_server_prep = f"{self.dir_base}/server_prep"

        self.mkdirs()

    def mkdirs(self):
        super().mkdirs()

        Path(self.dir_stats).mkdir(parents=True, exist_ok=True)

    # calculates some statistics on genetics and overall dataset
    def result_statistics(self, records):
        self.total_statistics(records)
        self.calculate_variation()

    def total_statistics(self, records):
        with open(f"{self.dir_stats}/total_stats.txt", 'w') as f:
            f.write(f"Number of entries: {len(records)}\n")
            f.write(f"Number of species: {len(records['species_name'].unique())}\n")
            for species in sorted(records['species_name'].unique()):
                f.write(f"{species}\n")

    def calculate_variation(self):
        print("Calculating the distance matrix...")
        self.r_genetic_distances()
        print("Converting distance matrix to long format...")
        self.r_output_to_long_format()

    def r_genetic_distances(self):
        alignment_f = f"{self.dir_server_prep}/{self.marker}.aln"
        distance_matrix_f = f"{self.dir_stats}/{self.marker}_distance_matrix.tsv"

        if os.path.isfile(distance_matrix_f):
            return

        p = subprocess.Popen(["Rscript", "--vanilla",
                              f"{self.script_dir}/R/genetic_distance.R",
                              "-a", alignment_f,
                              "-o", distance_matrix_f],
                             cwd=os.getcwd(),
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        _output, error = p.communicate()

        if not os.path.isfile(distance_matrix_f):
            print(alignment_f)
            print(distance_matrix_f)
            print(error.decode("utf-8"))
            sys.exit()

    # transforms distance matrix into long format
    def r_output_to_long_format(self):
        if os.path.isfile(f"{self.dir_stats}/{self.marker}_long_distances.tsv"):
            genetic_distances_long = pd.read_csv(f"{self.dir_stats}/{self.marker}_long_distances.tsv",
                                                 header=0, sep='\t', index_col=0)
        else:
            genetic_distances = pd.read_csv(f"{self.dir_stats}/{self.marker}_distance_matrix.tsv",
                                            header=0, sep='\t', index_col=0)

            genetic_distances.reset_index(level=0, inplace=True)
            genetic_distances.rename(columns={'index': 'query'}, inplace=True)
            records_ids = genetic_distances.columns[1:]

            combined_records_subset = pd.read_csv(f"{self.dir_merged_recs}/{self.job_id}_{self.marker}.tsv",
                                                  header=0, sep='\t')
            record_id_to_species = self.retrieve_species(records_ids, combined_records_subset)
            genetic_distances_long = self.wide_to_long_parallel(record_id_to_species, genetic_distances)

        intraspecific_rows = genetic_distances_long['query_species'] == genetic_distances_long['target_species']
        genetic_distances_long.loc[intraspecific_rows, 'species_comp_level'] = 'intraspecific'
        genetic_distances_long.loc[~intraspecific_rows, 'species_comp_level'] = 'interspecific'
        genetic_distances_long.to_csv(f"{self.dir_stats}/{self.marker}_long_distances.tsv",
                                      header=True, sep='\t', index=False)

    def wide_to_long_parallel(self, record_id_to_species, genetic_distances):
        records_ids = list(record_id_to_species.keys())
        chunk_size = int(len(records_ids) / 10)
        records_ids_chunks = [records_ids[i:i + chunk_size] for i in range(0, len(records_ids), chunk_size)]

        with mp.Pool(self.num_processes) as pool:
            record_id_records_list = pool.starmap(self.wide_to_long_ids, zip(records_ids_chunks,
                                                                             repeat(genetic_distances),
                                                                             repeat(record_id_to_species)))

        records = pd.concat(record_id_records_list, ignore_index=True)

        species_to_id = defaultdict(list)
        for key, value in sorted(record_id_to_species.items()):
            species_to_id[value].append(key)

        for species_name, ids in species_to_id.items():
            records.loc[records['query'].isin(ids), 'query_species'] = species_name

        return records

    def wide_to_long_ids(self, records_ids, genetic_distances, record_id_to_species):
        record_id_records = [self.wide_to_long_id(genetic_distances[['query', record_id]].copy(),
                                                  record_id, record_id_to_species) for
                             record_id in records_ids]

        return pd.concat(record_id_records, ignore_index=True)

    def wide_to_long_id(self, records, record_id, record_id_to_species):
        records['target'] = record_id
        records['target_species'] = record_id_to_species[record_id]
        records.rename(columns={record_id: 'distance'}, inplace=True)
        return records

    def retrieve_species(self, record_ids, combined_records_subset):
        record_id_to_species = {}

        for record_id in record_ids:
            record = combined_records_subset[combined_records_subset['record_id'] == record_id]

            species_name = record['species_name'].values[0]
            record_id_to_species[record_id] = species_name

        return record_id_to_species
