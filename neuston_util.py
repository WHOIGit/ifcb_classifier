#!/usr/bin/env
"""This module exists to run useful auxiliary neuston tasks"""
import argparse
import os
import csv


def write_csv(outfile, rows):
    if outfile:
        with open(args.outfile,'w') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    else:
        for row in rows:
            print(','.join(row))


def make_dataset_config(args):

    # parsing datasets
    datasets = []
    priorities = []
    for src in args.dataset:
        src = src.split(':',1)
        if len(src)==2:
            datasets.append(src[1])
            priorities.append(int(src[0]))
        else:
            datasets.append(src[0])
            priorities.append(0)
    priorities = [p if p>0 else max(priorities)+1 for p in priorities]

    # collecting classes for each dataset
    classes = set()
    dataset_subdirs = []
    for dataset in datasets:
        subdirs = [subdir for subdir in os.listdir(dataset) if os.path.isdir(os.path.join(dataset,subdir))]
        dataset_subdirs.append(subdirs)
        classes.update(subdirs)
    classes = sorted(classes)

    # creating csv data
    header = ['']+['{}:{}'.format(p, d) for p, d in zip(priorities, datasets)]
    rows = []
    for cls in classes:
        defaults = ['1' if cls in dssd else '0' for dssd in dataset_subdirs]
        row = [cls] + defaults
        rows.append(row)
    write_csv(args.outfile,[header]+rows)



def make_class_config(args):

    # fetch classes
    if os.path.isdir(args.dataset):
        classes = [subdir for subdir in os.listdir(args.dataset) if os.path.isdir(os.path.join(args.dataset,subdir))]
    elif os.path.isfile(args.dataset) and args.dataset.endswith('.csv'):
        with open(args.dataset) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        classes = [row[0] for row in rows if any([val!='0' for val in row[1:]])]

    # creating csv data
    header = [args.dataset,'CONFIG1']
    rows=[]
    for cls in classes:
        rows.append([cls,'1'])
    write_csv(args.outfile,[header]+rows)


def main(args):
    if args.cmd=='MAKE_DATASET_CONFIG':
        make_dataset_config(args)
    elif args.cmd=='MAKE_CLASS_CONFIG':
        make_class_config(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd', help='These sub-commands are mutually exclusive.')

    # DATASET CONFIG CSV #
    dataset_config = subparsers.add_parser('MAKE_DATASET_CONFIG', help='Creates a default dataset-combining configuration file.')
    dataset_config.add_argument('dataset', metavar='PATH', nargs='+',
                                help='List of dataset paths. Space deliminated. '
                                     'You may optionally prefix the paths with "n:" where n is an integer priority value. Lower values are higher priority.'
                                     'Multiple Datasets may have the same priority level. '
                                     'If only some datasets have priority values, datasets without priority values are designated with the lowers priority level.')
    dataset_config.add_argument('-o', '--outfile', help='Specify an output file. If unset, outputs to stdout.')

    # CLASS-CONFIG CSV #
    class_config = subparsers.add_parser('MAKE_CLASS_CONFIG', help='Creates a default class-config csv file.')
    class_config.add_argument('dataset',metavar='PATH', help='path to a dataset directory or dataset configuration csv file.')
    class_config.add_argument('-o', '--outfile', help='Specify an output file. If unset, outputs to stdout.')

    # run util command
    args = parser.parse_args()
    main(args)

