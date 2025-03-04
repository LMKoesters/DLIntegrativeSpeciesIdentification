import argparse
from glob import glob
import os
from pathlib import Path
import re
import sys

from train_parent import CrossValidation, TraditionalTraining


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", "-j", help="set job id for folder name (required)", required=True)
    parser.add_argument("--num-workers", "-w", help="workers for data loading (default: 4)", type=int, default=4)
    parser.add_argument("--num-epochs", "-o", help="number of epochs to train (default: 500)", type=int, default=500)
    parser.add_argument("--run-index", "-i", help="set subset index (0-based; same machine)", type=int, nargs='?',
                        default=0)
    parser.add_argument("--runs", "-r", help="set number of subsets to run in parallel (same machine)", type=int,
                        nargs='?', default=1)
    parser.add_argument("--batches", "-x", type=int, nargs='?', default=1,
                        help='set subset index (0-based; different machines)')
    parser.add_argument("--batch-num", "-y", type=int, nargs='?', default=0,
                        help='set number of subsets to run in parallel (different machines)')
    parser.add_argument("--hide-progress", '-z', default=False, action=argparse.BooleanOptionalAction,
                        help='hide progress bars')
    parser.add_argument("--subset", '-s', default=False, action=argparse.BooleanOptionalAction,
                        help='use dataset subset for LOOCV (default: False)')
    parser.add_argument("--root-dir", '-d', nargs='?', default='../..',
                        help='Folder in which data folder is located (default: grandparent directory)')
    parser.add_argument("--cv-folds", '-f', nargs='?', default=0, type=int,
                        help='number of folds when applying cross validation (default: 0 = leave one out)')
    parser.add_argument("--classifiers", '-c', nargs='*', default=['product_score', 'sum_score',
                                                                   'max_score', 'dense_mid', 'dense_late', 'rf'])
    parser.add_argument("--processings", '-p', nargs='*',
                        default=['aligned_barcode', 'aligned_barcode_snp', 'padded_barcode'])
    parser.add_argument("--encodings", '-e', nargs='*', default=['one_hot_bar', 'sequential_bar'])
    parser.add_argument("--pretrainings", '-g', nargs='*', default=['sep'])
    parser.add_argument("--traditional", '-t', default=True, action=argparse.BooleanOptionalAction,
                        help='train with traditional train-val split (default: True)')
    parser.add_argument("--cross-validation", '-v', default=True, action=argparse.BooleanOptionalAction,
                        help='apply cross validation (default: True)')
    parser.add_argument("--blast", '-b', default=False, action=argparse.BooleanOptionalAction,
                        help='run BLAST evaluation for dataset (default: False)')
    return parser.parse_args(args=None if sys.argv[1:] else ['--help'])


if __name__ == '__main__':
    args = get_args()
    job_id = args.job_id

    # set up results directory
    if args.cross_validation:
        if args.cv_folds == 0:
            results_dir = f'{args.root_dir}/results/{job_id}/loocv'
        else:
            results_dir = f'{args.root_dir}/results/{job_id}/cv_{args.cv_folds}_folds'
    else:
        results_dir = f'{args.root_dir}/results/{job_id}/traditional'

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # remove all work in progress files of batches that are not running anymore
    for f in glob(f'{results_dir}/wips_*.tsv'):
        if int(re.search(r"(?<=wips_)\d+", f).group(0)) >= args.runs:
            os.remove(f)
    # remove the work in progress file for this batch
    Path(f'{results_dir}/wips_{args.run_index}.tsv').unlink(missing_ok=True)

    # initialize and run cross validation/traditional training
    if args.cross_validation:
        cv = CrossValidation(results_dir, True, args)
        bar_enc, bar_prep = cv.run_preprocessing()

        args.encodings = [bar_enc]
        args.processings = [bar_prep]
        TraditionalTraining(f'{args.root_dir}/results/{job_id}/traditional', False, args).run()
        cv.run_fusion(args.encodings, args.processings)
    else:
        TraditionalTraining(results_dir, False, args).run()
