#!/usr/bin/env python

import argparse
from ast import literal_eval
import numpy as np
from scipy.io import savemat


def read_epoch(fpath):
    with open(fpath, 'r') as f:
        epoch = literal_eval(f.read())
    return epoch


def save_as_mat(fpath, input_classes, input_rois, output_classes, output_ranks, classes):
    assert len(input_classes)==len(input_rois)==len(output_classes)==len(output_ranks)

    d = dict(classnames = np.asarray(classes, dtype='object'),
             input_rois = np.asarray(input_rois, dtype='object'),
             input_classes = np.array(input_classes),
             output_classes = np.array(output_classes),
             output_ranks = np.array(output_ranks).astype('f4'),
             )

    savemat(fpath, d, do_compression=True)


def main(infile, outfile):
    ep = read_epoch(infile)
    input_rois = [img.split('/')[-1].rsplit('.',1)[0] for img in ep['true_images']]
    params = dict(input_rois = input_rois,
                  input_classes = ep['true_inputs'],
                  output_classes = ep['prediction_outputs'],
                  output_ranks = ep['prediction_ranks'],
                  classes = ep['classes'],
                  )
    save_as_mat(outfile,**params)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='input filepath. Eg: path/to/best_epoch.dict')
    parser.add_argument('outfile', help='output filepath. Eg: path/to/yourfilename.mat')
    args = parser.parse_args()

    main(args.infile, args.outfile)

