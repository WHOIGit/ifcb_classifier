"""make new datasets"""


from subprocess import run, PIPE
from random import shuffle
import argparse, os
import shutil

# parse CLI arguments
parser = argparse.ArgumentParser(description='recombines a dataset into a new datasetset')
parser.add_argument('--src', default = 'data/ifcb_fixed_fullset', help = 'directory to source data from')
parser.add_argument('--dst', help = 'directory to move files over to')
parser.add_argument('--num-o-classes', '--noc', type=int, help='number of classes to copy over, classes with the most images prioritized')
parser.add_argument('--split', nargs=2, type=int, help='splits source subset into training and validation sets. eg: "--split 80 20" for 80%% training, 20%% validation)')
parser.add_argument('--exclude', nargs='+', help='exclude these classes from the output')
parser.add_argument('--specific-dirs', nargs='+', help='specify exactly which classes to transcribe (default all)') ##TODO wildcards
args = parser.parse_args()

# which species to copy
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
    subdirs.append(dict(species=subdir,files=file_list))

# filter out any species to exclude
if args.exclude:
    assert set(args.exclude).issubset(subdir_list)
    for dump_dir in args.exclude:
        subdirs = [subdir for subdir in subdirs if subdir['species'] != dump_dir]

# sort by classes with the most files and prune to the top x classes
subdirs.sort(key=lambda x:len(x['files']),reverse=True)
if args.num_o_classes is not None:
    subdirs = subdirs[:args.num_o_classes]

# make lists of files for testing and training
if args.split:
    for subdir in subdirs:
        cuttoff = round(len(subdir['files']) * args.split[0]/100)
        shuffle(subdir['files'])  # shuffled to make avoid bias
        subdir['training_files'] = subdir['files'][:cuttoff]
        subdir['validation_files'] = subdir['files'][cuttoff:]
        outstr = '{:>30}    {} files:    {} Training    {} Validating'
        print(outstr.format(subdir['species'],len(subdir['files']),len(subdir['training_files']),len(subdir['validation_files'])))
else:
    for subdir in subdirs:
        outstr = '{:>30}    {} files'
        print(outstr.format(subdir['species'], len(subdir['files'])))
print('Summary: {} classes, {} total files'.format(len(subdirs),sum([len(subdir['files']) for subdir in subdirs])))

# copying of files
training = 'training'
validation = 'validation'
if args.dst:
    copy_confirm = input('are you sure you want to copy these to "{}" ? Y/n '.format(args.dst))
    if copy_confirm in ['Y','yes','YES','Yes']:
        os.mkdir(args.dst)
        if args.split:
            os.mkdir(os.path.join(args.dst, training))
            os.mkdir(os.path.join(args.dst, validation))
            for subdir in subdirs:

                print('Copying {}...'.format(subdir['species']))
                # copying training files
                os.mkdir(os.path.join(args.dst, training, subdir['species']))
                for f in subdir['training_files']:
                    src = os.path.join(args.src, subdir['species'], f)
                    dst = os.path.join(args.dst, training, subdir['species'], f)
                    shutil.copyfile(src,dst)
                os.mkdir(os.path.join(args.dst, validation, subdir['species']))
                # copying validation files
                for f in subdir['validation_files']:
                    src = os.path.join(args.src, subdir['species'], f)
                    dst = os.path.join(args.dst, validation, subdir['species'], f)
                    shutil.copyfile(src,dst)
        else:
            for subdir in subdirs:
                # copy all files
                print('Copying {}...'.format(subdir['species']))
                os.makedirs(os.path.join(args.dst, subdir['species']))
                for f in subdir['files']:
                    src = os.path.join(args.src, subdir['species'], f)
                    dst = os.path.join(args.dst, subdir['species'], f)
                    shutil.copyfile(src,dst)


