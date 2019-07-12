#! /usr/bin/env python
"""make new datasets"""

from subprocess import run, PIPE
from random import shuffle
import argparse, os
import shutil, fnmatch

# parse CLI arguments
parser = argparse.ArgumentParser(description='recombines a dataset into a new datasetset')
parser.add_argument('--src', help = 'directory to source data from')
parser.add_argument('--dst', help = 'directory to move files over to')
parser.add_argument('--num-o-classes', '--noc', type=int, help='number of classes to copy over, classes with the most images prioritized')
parser.add_argument('--split', default='', help='''splits source subset into training and validation sets.
eg: "--split 80:20" for 80%% training, 20%% validation, or...
eg: "-split :2018-10-20*" to move files with names that start 
     with that particluar timstamps from training dir to testing dir''')
parser.add_argument('--exclude', nargs='+', help='exclude these classes from the output')
parser.add_argument('--specific-dirs', nargs='+', help='specify exactly which classes to transcribe (default all)') ##TODO wildcards
parser.add_argument('--filematch', default=False, help='only include files that match filematch. * and ? wildcards available (default *)')
parser.add_argument('--silent',default=False, action='store_true', help='skip confirmation step if --dst is used')
args = parser.parse_args()

# which dirname to copy
if args.specific_dirs:
    assert set(args.specific_dirs).issubset(os.listdir(args.src))
    subdir_list = args.specific_dirs
else:
    subdir_list = os.listdir(args.src)

# make file lists
subdirs = []
for subdir in subdir_list:
    #cmd = ['ls', os.path.join(args.src,d)]
    #fcount = run(cmd,stdout=PIPE).stdout.decode().strip().split('\n')
    #cmd = ['find', os.path.join(args.src,subdir), '-type', 'f']
    #flist = run(cmd,stdout=PIPE).stdout.decode().strip().split('\n')
    #flist = [f.rsplit('/',1)[-1] for f in flist]
    file_list = os.listdir(os.path.join(args.src,subdir))
    if args.filematch:
        file_list = fnmatch.filter(file_list,args.filematch)
    subdirs.append(dict(dirname=subdir,files=file_list))

# filter out any dirnames to exclude
if args.exclude:
    assert set(args.exclude).issubset(subdir_list)
    for dump_dir in args.exclude:
        subdirs = [subdir for subdir in subdirs if subdir['dirname'] != dump_dir]

# sort by classes with the most files and prune to the top x classes
subdirs.sort(key=lambda x:len(x['files']),reverse=True)
if args.num_o_classes is not None:
    subdirs = subdirs[:args.num_o_classes]

# make lists of files for testing and training
if ':' in args.split:
    training_split = args.split.split(':')[0]
    validation_split = args.split.split(':')[1]
    try: training_split = int(training_split)
    except ValueError:
        try: training_split = 100-int(validation_split)
        except ValueError: pass
    for subdir in subdirs:
        if isinstance(training_split,int):
            cuttoff = round(len(subdir['files']) * training_split/100)
            shuffle(subdir['files'])  # shuffled to make avoid bias
            subdir['training_files'] = subdir['files'][:cuttoff]
            subdir['validation_files'] = subdir['files'][cuttoff:]
        elif validation_split:
            subdir['validation_files'] = fnmatch.filter(subdir['files'],validation_split)
            subdir['training_files'] = [f for f in subdir['files'] if f not in subdir['validation_files']]
        elif training_split:
            subdir['training_files'] = fnmatch.filter(subdir['files'],training_split)
            subdir['validation_files'] = [f for f in subdir['files'] if f not in subdir['training_files']]
        else:
            print('--split {} could not be parsed'.format(args.split))
        outstr = '{:>30}    {} files:    {} Training    {} Validating'
        print(outstr.format(subdir['dirname'], len(subdir['files']), len(subdir['training_files']), len(subdir['validation_files'])))
else:
    for subdir in subdirs:
        outstr = '{:>30}    {} files'
        print(outstr.format(subdir['dirname'], len(subdir['files'])))
print('Summary: {} classes, {} total files'.format(len(subdirs),sum([len(subdir['files']) for subdir in subdirs])))

# copying of files
training = 'train'
validation = 'test'
if args.dst:
    if args.silent == True:
        copy_confirm = 'yes'
    else:
        copy_confirm = input('are you sure you want to copy these to "{}" ? Y/n '.format(args.dst))
    if copy_confirm.lower() in ['y','yes']:
        os.mkdir(args.dst)
        if args.split:
            os.mkdir(os.path.join(args.dst, training))
            os.mkdir(os.path.join(args.dst, validation))
            for subdir in subdirs:

                print('Copying {}...'.format(subdir['dirname']))
                # copying training files
                os.mkdir(os.path.join(args.dst, training, subdir['dirname']))
                for f in subdir['training_files']:
                    src = os.path.join(args.src, subdir['dirname'], f)
                    dst = os.path.join(args.dst, training, subdir['dirname'], f)
                    shutil.copyfile(src,dst)
                os.mkdir(os.path.join(args.dst, validation, subdir['dirname']))
                # copying validation files
                for f in subdir['validation_files']:
                    src = os.path.join(args.src, subdir['dirname'], f)
                    dst = os.path.join(args.dst, validation, subdir['dirname'], f)
                    shutil.copyfile(src,dst)
        else:
            for subdir in subdirs:
                # copy all files
                print('Copying {}...'.format(subdir['dirname']))
                os.makedirs(os.path.join(args.dst, subdir['dirname']))
                for f in subdir['files']:
                    src = os.path.join(args.src, subdir['dirname'], f)
                    dst = os.path.join(args.dst, subdir['dirname'], f)
                    shutil.copyfile(src,dst)

