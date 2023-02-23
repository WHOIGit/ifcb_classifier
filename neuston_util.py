#!/usr/bin/env
"""This module exists to run useful auxiliary neuston tasks"""
import argparse
import os
import csv

import numpy as np

from neuston_data import NeustonDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def calc_img_norm(args):

    tforms=transforms.Compose([transforms.Resize(2*[args.resize]),transforms.ToTensor()])

    if not args.class_config:
        nd = NeustonDataset(src=args.SRC, transforms=tforms,
                            minimum_images_per_class=args.class_min, maximum_images_per_class=args.class_max)
    else:
        nd = NeustonDataset.from_csv(src=args.SRC, transforms=tforms,
                                     csv_file=args.class_config[0], column_to_run=args.class_config[1],
                                     minimum_images_per_class=args.class_min, maximum_images_per_class=args.class_max)
    dataloader = DataLoader(nd, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_batches = len(dataloader)

    pop_mean = []
    pop_std0 = []
    for i,data in enumerate(dataloader,1):
        img_data,_,_ = data
        # shape (batch_size, 3, height, width)
        numpy_image = img_data.numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
        #batch_std1 = np.std(numpy_image, axis=(0, 2, 3), ddof=1)

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)

        if i%100==0:
            line = '\n{:.1f}% ({} of {}) MEAN={} STD={}'
            line = line.format(100*i/num_batches,i,num_batches,
                               np.array(pop_mean).mean(axis=0)[0],
                               np.array(pop_std0).mean(axis=0)[0])
            print(line)
        else:
            print('.',end='',flush=True)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    mean = np.array(pop_mean).mean(axis=0)
    std0 = np.array(pop_std0).mean(axis=0)
    return mean,std0

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
    else:
        raise ValueError(f'Dataset is invalid: "{args.dataset}"')
    classes.sort()

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
    elif args.cmd=='CALC_IMG_NORM':
        print('Calculating Image Normalization MEAN and STD...')
        mean,std = calc_img_norm(args)
        print('MEAN={}, STD={}'.format(mean,std))


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

    # IMAGE NORMALIZATION
    imgnorm = subparsers.add_parser('CALC_IMG_NORM', help='Calculate the MEAN and STD of dataset for image normalizing')
    imgnorm.add_argument('SRC')
    imgnorm.add_argument('--resize', metavar='N', default=299, type=int, choices=[224,299], help='Default is 299 (for inception_v3)')
    imgnorm.add_argument('--class-config', metavar=('CSV', 'COL'), nargs=2, help='Skip and combine classes as defined by column COL of a special CSV configuration file')
    imgnorm.add_argument('--class-min', metavar='MIN', default=2, type=int, help='Exclude classes with fewer than MIN instances. Default is 2')
    imgnorm.add_argument('--class-max', metavar='MAX', default=None, type=int, help='Limit classes to a MAX number of instances. '
                           'If multiple datasets are specified with a dataset-configuration csv, classes from lower-priority datasets are truncated first.')
    imgnorm.add_argument('--batch-size', metavar='B', default=108, help='Number of images per minibatch')

    # run util command
    args = parser.parse_args()
    main(args)

