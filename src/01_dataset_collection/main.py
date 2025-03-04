import argparse
import os
from pathlib import Path

from dataset import Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", "-j", help="set job id for folder name (default: same as query)")
    parser.add_argument("--query", "-q", help="set query term to search for")
    parser.add_argument("--query-type", "-t", help="set query type (default: taxon)",
                        default='taxon',
                        choices=['taxon', 'container', 'accessions'])
    parser.add_argument("--accessions", '-a', help="tsv with GenBank accessions",
                        default=None)
    parser.add_argument("--num-processes", "-n", help="set number of cores to use",
                        type=int,
                        default=8)
    parser.add_argument("--threshold", '-d', help="set minimum number of entries/species+marker",
                        type=int,
                        default=4)
    parser.add_argument("--min-species", '-m', help="set minimum number of species in dataset",
                        type=int,
                        default=40)
    parser.add_argument("--project-dir", '-p',
                        default=Path(os.path.dirname(os.path.realpath(__file__))).parent.parent,
                        help='The directory of the project. The result directory is created within this directory.')
    parser.add_argument("--fresh-images", default=False, action='store_true',
                        help="Switch to human observations instead of preserved specimens")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    if not args.job_id:
        args.job_id = args.query

    Dataset(args).compile()
