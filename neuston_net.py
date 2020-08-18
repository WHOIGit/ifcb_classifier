#!/usr/bin/env python
"""the main thing"""

# built in imports
from shutil import copyfile
import argparse
import os, glob

# 3rd party imports
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision.datasets.folder import IMG_EXTENSIONS

# project imports
import ifcb
from neuston_models import NeustonModel
from neuston_callbacks import SaveValidationResults, SaveRunResults
from csv_logger import CSVLogger  # ptl_v0.8 does not have this but ptl_v0.9 does
from neuston_data import get_trainval_datasets, IfcbBinDataset, ImageDataset

## NOTES ##
# https://pytorch-lightning.readthedocs.io/en/0.8.5/introduction_guide.html


def main(args):

    ## GPU torch setup ## - use all available GPU's
    # pytorch uses the ~index~ of CUDA_VISIBLE_DEVICES.
    # So if gpus == [3,4] then device "cuda:0" == GPU no. 3
    #                      and device "cuda:1" == GPU no. 4
    if torch.cuda.is_available():
        args.gpus = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else: args.gpus = None

    if args.cmd_mode=='TRAIN':
        do_training(args)
    else: # RUN
        do_run(args)


def do_training(args):

    # Setup Callbacks
    validation_results_callbacks = []
    plotting_callbacks = [] # TODO
    if not args.result_files: args.result_files = [['results.json','image_basenames']]
    for result_file in args.result_files:
        svr = SaveValidationResults(outdir=args.OUTDIR, outfile=result_file[0], series=result_file[1:])
        validation_results_callbacks.append(svr)

    # Set Seed. If args.seed is 0 ie None, a random seed value is used and stored
    args.seed = seed_everything(args.seed or None)

    #if os.path.isfile(args.MODEL): #TODO: transfer learning option
    # see https://pytorch-lightning.readthedocs.io/en/stable/transfer_learning.html?highlight=predictions

    # Setup dataloaders
    training_dataset, validation_dataset = get_trainval_datasets(args)
    assert training_dataset.classes == validation_dataset.classes
    args.classes = training_dataset.classes
    # TODO add to args classes removed by class_min and skipped/combined from class_config

    print('Loading Training Dataloader...')
    training_loader = DataLoader(training_dataset, pin_memory=True, shuffle=True,
                                 batch_size=args.batch_size, num_workers=args.loaders)
    print('Loading Validation Dataloader...')
    validation_loader = DataLoader(validation_dataset, pin_memory=True, shuffle=False,
                                   batch_size=args.batch_size, num_workers=args.loaders)

    # Setup Trainer
    logger = CSVLogger(save_dir=os.path.join(args.OUTDIR,'logs'), name='default', version=None)
    chkpt_path = os.path.join(args.OUTDIR, 'chkpts')
    os.makedirs(chkpt_path, exist_ok=True)
    trainer = Trainer(deterministic=True, logger=logger,
                      gpus=len(args.gpus) if args.gpus else None,
                      max_epochs=args.emax, min_epochs=args.emin,
                      early_stop_callback=EarlyStopping(patience=args.estop) if args.estop else False,
                      checkpoint_callback=ModelCheckpoint(filepath=chkpt_path),
                      callbacks=validation_results_callbacks,
                      )

    # Setup Model
    classifier = NeustonModel(args)
    # TODO setup dataloaders in the model, allowing auto-batch-size optimization
    # see https://pytorch-lightning.readthedocs.io/en/stable/training_tricks.html#auto-scaling-of-batch-size

    # Do Training
    trainer.fit(classifier, train_dataloader=training_loader, val_dataloaders=validation_loader)

    # Copy best model
    checkpoint_path = trainer.checkpoint_callback.best_model_path
    output_path = os.path.join(args.OUTDIR, args.model_file)
    copyfile(checkpoint_path, output_path)

    # Copying Logs
    if args.epochs_log:
        output_path = os.path.join(args.OUTDIR, args.epochs_log)
        copyfile(logger.experiment.metrics_file_path, output_path)
    if args.args_log:
        src_path = os.path.join(logger.experiment.log_dir, logger.experiment.NAME_HPARAMS_FILE)
        output_path = os.path.join(args.OUTDIR, args.args_log)
        copyfile(src_path, output_path)


def do_run(args):

    # load model
    classifier = NeustonModel.load_from_checkpoint(args.MODEL)
    classifier.args.run_outfile = args.outfile
    classifier.args.run_outdir = args.OUTDIR
    seed_everything(classifier.args.seed)

    # Setup Callbacks
    run_results_callbacks = []
    plotting_callbacks = []  # TODO
    #for result_file in args.result_files:
    #    svr = SaveRunResults(outdir=args.OUTDIR, outfile=result_file[0], series=result_file[1:])
    #    run_results_callbacks.append(svr)

    # create trainer
    trainer = Trainer(deterministic=True,
                      gpus=len(args.gpus) if args.gpus else None,
                      logger=False, checkpoint_callback=False,
                      callbacks=run_results_callbacks,
                      )

    # dataset filter if any
    filter_mode, filter_keywords = None,[]
    if args.filter:
        filter_mode = args.filter[0]
        for keyword in args.filter[1:]:
            if os.path.isfile(keyword):
                with open(keyword) as f:
                    filter_keywords.extend(f.readlines())
            else:
                filter_keywords.append(keyword)

    # create dataset
    if args.src_type == 'bin':
        # Formatting Dataset
        if os.path.isdir(args.SRC):
            if filter_mode=='IN':
                dd = ifcb.DataDirectory(args.SRC, whitelist=filter_keywords)
            elif filter_mode=='OUT':
                dd = ifcb.DataDirectory(args.SRC, blacklist=filter_keywords)
            else:
                dd = ifcb.DataDirectory(args.SRC)
        elif os.path.isfile(args.SRC) and args.SRC.endwith('.txt'): # TODO TEST: textfile bin run
            with open(args.SRC,'r') as f:
                bins = f.readlines()
            parent = os.path.commonpath(bins)
            dd = ifcb.DataDirectory(parent,whitelist=bins)
        else: # single bin # TODO TEST: single bin run
            parent = os.path.dirname(args.SRC)
            bin_id = os.path.basename(args.src)
            dd = ifcb.DataDirectory(parent,whitelist=[bin_id])

        for i, bin_id in enumerate(dd):

            if args.filter: # applying filter
                if filter_mode=='IN': # if bin does NOT match any of the keywords, skip it
                    if not any([k in str(bin_id) for k in filter_keywords]): continue
                elif filter_mode=='OUT': # if bin matches any of the keywords, skip it
                    if any([k in str(bin_id) for k in filter_keywords]): continue

            bin_dataset = IfcbBinDataset(bin_id, classifier.args.resize)
            image_loader = DataLoader(bin_dataset, batch_size=args.batch_size,
                                      pin_memory=True, num_workers=args.loaders)

            # skip empty bins
            if len(image_loader) == 0: continue

            # Do Runs
            trainer.test(classifier, test_dataloaders=image_loader)

    else: # images
        img_paths = []
        if os.path.isdir(args.SRC):
            for img in glob.iglob(os.path.join(args.SRC,'**','**'), recursive=True):
                if any([img.endswith(ext) for ext in IMG_EXTENSIONS]):
                    img_paths.append(img)
        elif os.path.isfile(args.SRC) and args.SRC.endwith('.txt'): # TODO TEST: textfile img run
            with open(args.SRC,'r') as f:
                img_paths = f.readlines()
        elif any([args.SRC.endswith(ext) for ext in IMG_EXTENSIONS]): # single img # TODO TEST: single img run
            img_paths.append(args.SRC)

        # applying filter
        if args.filter:
            for img in img_paths[:]:
                if filter_mode=='IN': # if img does NOT match any of the keywords, skip it
                    if not any([k in img for k in filter_keywords]): img_paths.remove(img)
                elif filter_mode=='OUT': # if img matches any of the keywords, skip it
                    if any([k in img for k in filter_keywords]): img_paths.remove(img)

        assert len(img_paths)>0, 'No images to process'
        image_dataset = ImageDataset(img_paths, resize=classifier.args.resize)
        image_loader = DataLoader(image_dataset, batch_size=args.batch_size,
                                  pin_memory=True, num_workers=args.loaders)

        trainer.test(classifier,test_dataloaders=image_loader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train, Run, and perform other tasks related to ifcb and general image classification!')

    # Create subparsers
    subparsers = parser.add_subparsers(dest='cmd_mode', help='These sub-commands are mutually exclusive. Note: optional arguments (below) must be specified before "TRAIN" or "RUN"')
    train = subparsers.add_parser('TRAIN', help='Train a new model')
    run = subparsers.add_parser('RUN', help='Run a previously trained model')
    # TODO the below to be included later
    #check = subparsers.add_parser('check', help='Run and compare a trained model against its validation dataset results')
    #dupes = subparsers.add_parser('dupes', help='Perform a special training') # this may be best suited to it's own "test-tube" based HPC-enabled module.
    # TODO the below may be better suited to a utils module
    #norm = subparsers.add_parser('norm', help='Determine the MEAN and STD values to use for --img-norm')
    #config= subparsers.add_parser('classconfig', help='create a default class-config ready csv for a given dataset')

    ## Common Vars ##
    parser.add_argument('--batch', dest='batch_size', metavar='SIZE', default=108, type=int, help='Number of images per batch. Defaults is 108') # todo: auto-mode built in to ptl
    parser.add_argument('--loaders', metavar='N', default=4, type=int, help='Number of data-loading threads. 4 per GPU is typical. Default is 4') # todo: auto-mode?

    ## Training Vars ##
    train.add_argument('MODEL', help='Select a base model. Eg: "inception_v3"') # TODO choices field. TODO: "Accepts a known model name, or a path to a specific model file for transfer learning"
    train.add_argument('SRC', help='Directory with class-label subfolders and images')
    train.add_argument('OUTDIR', help='Set output directory.')

    model = train.add_argument_group(title='Model Adjustments', description=None)
    model.add_argument('--untrain', dest='pretrained', default=True, action='store_false', help='If set, initializes MODEL ~without~ pretrained neurons. Default (unset) is pretrained')
    model.add_argument('--img-norm', nargs=2, metavar=('MEAN','STD'), help='Normalize images by MEAN and STD. This is like whitebalancing.')
    # TODO layer freezing and transfer learning params.

    data = train.add_argument_group(title='Dataset Adjustments', description=None)
    data.add_argument('--seed', default=0, type=int, help='Set a specific seed for deterministic output & dataset-splitting reproducability.')
    data.add_argument('--split', metavar='T:V', default='80:20', help='Ratio of images per-class to split randomly into Training and Validation datasets. Randomness affected by SEED. Default is "80:20"')
    data.add_argument('--class-config', metavar=('CSV','COL'), nargs=2, help='Skip and combine classes as defined by column COL of a special CSV configuration file')
    data.add_argument('--class-min', metavar='MIN', default=2, type=int, help='Exclude classes with fewer than MIN instances. Default is 2')
    #data.add_argument('--swap', action='store_true', help=argparse.SUPPRESS)  # dupes placeholder. may not be needed.
    
    epochs = train.add_argument_group(title='Epoch Parameters', description=None)
    epochs.add_argument('--emax', metavar='MAX',default=60, type=int, help='Maximum number of training epochs. Default is 60')
    epochs.add_argument('--emin', metavar='MIN', default=10, type=int, help='Minimum number of training epochs. Default is 10')
    epochs.add_argument('--estop', metavar='STOP', default=10, type=int, help='Early Stopping: Number of epochs following a best-epoch after-which to stop training. Set STOP=0 to disable. Default is 10')

    augs = train.add_argument_group(title='Augmentation Options', description='Data Augmentation is a technique by which training results may improved by simulating novel input')
    augs.add_argument('--flip', choices=['x','y','xy','x+V','y+V','xy+V'],
        help='Training images have 50%% chance of being flipped along the designated axis: (x) vertically, (y) horizontally, (xy) either/both. May optionally specify "+V" to include Validation dataset')

    out = train.add_argument_group(title='Output Options')
    out.add_argument('--model-file', default='model.ptl', help='The file name of the output model. Default is model.ptl')
    out.add_argument('--epochs-log', default='epochs.csv', help='Specify a csv filename. Includes epoch, loss, validation loss, and f1 scores. Default is epochs.csv')
    out.add_argument('--args-log', help='Specify a yaml filename. Includes all user-specified and default training parameters.')
    out.add_argument('--results',  dest='result_files', metavar=('FNAME','SERIES'), nargs='+', action='append', default=[],
                     help='FNAME: Specify a validation-results filename or pattern. Valid patterns are: "{epoch}". Accepts .json .h5 and .mat file formats. Default is "results.json". '
                          'SERIES: Options are: image_basenames, image_fullpaths, output_ranks, output_fullranks, confusion_matrix. class_labels, input_classes, output_classes are always included by default. '
                          '--results may be specified multiple times in order to create different files. If not invoked, default is "results.json image_basenames"')
    #out.add_argument('-p','--plot', metavar=('FNAME','PARAM'), nargs='+', action='append', help='Make Plots') # TODO plots

    #optim = train.add_argument_group(title='Optimization', description='Adjust learning hyper parameters')
    #optim.add_argument('--optimizer', default='Adam', choices=['Adam'], help='Select and optimizer. Default is Adam')
    #optim.add_argument('--learning-rate',default=0.001,type=float,help='Set a learning rate. Default is 0.001')
    #optim.add_argument('--weight-decay', default='?', help="not sure where this comes in")
    #optim.add_argument('--class-norm', help='Bias results to emphasize smaller classes')
    #optim.add_argument('--batch-norm', help='i forget what this is exactly rn')

    ## Run Vars ##
    run.add_argument('MODEL', help='Path to a previously-trained model file')
    run.add_argument('SRC', help='Resource(s) to be classified. Accepts a bin, an image, a text-file, or a directory. Directories are accessed recursively')
    run.add_argument('OUTDIR', help='Set output directory.')

    run.add_argument('--type', dest='src_type', default='bin', choices=['bin','img'], help='File type to perform classification on. Defaults is "bin"')
    run_outfile = run.add_argument('--outfile', default="{bin}_class_v2.h5",
        help='''Name/pattern of the output classification file. 
                If TYPE==bin, "{bin}" in OUTFILE will be replaced with the bin id on a per-bin basis.
                A few output file formats are recognized: .csv, .mat, and .h5 (hdf).
                Default for TYPE==bin is "{bin}_class_v2.h5"; Default for TYPE==img is "img_results.csv".
             ''') # TODO? If TYPE==img, "{dir}" in OUTFILE will be replaced with the parent directory of classified images."
    run.add_argument('--filter', nargs='+', metavar=('IN|OUT','KEYWORD'),
        help='Explicitly include (IN) or exclude (OUT) bins or image-files by KEYWORDs. KEYWORD may also be a text file containing KEYWORDs, line-deliminated.')
    #run.add_argument('-p','--plot', metavar=('FNAME','PARAM'), nargs='+', action='append', help='Make Plots') # TODO plots

    args = parser.parse_args()

    # CORRECTIONS AND CHECKS
    if args.cmd_mode=='RUN':
        # change default text of OUTFILE if TYPE==img
        if args.src_type=='img' and args.outfile==run_outfile.default:
            args.outfile = 'img_results.csv'
        if args.filter:
            if not args.filter[0] in ['IN','OUT']:
                argparse.ArgumentTypeError('IN|OUT must be either "IN" or "OUT"')
            if len(args.filter)<2:
                argparse.ArgumentTypeError('Must be at least one KEYWORD')

    main(args)

# TODO (minor) run outfile using callbacks. better flexibility
# TODO (minor) move dataloaders to NeustonModel for auto-batch-size enabling
# TODO (minor) utility script to fine img-norm MEAN STD
# TODO (minor) bin script for connecting to hpc, connecting to a gpu, enabling conda env, and moving to working directory
# TODO run on larger, current dataset using class-config
# TODO implement running on images
# TODO implement plots (matplotlib vs plotly?)
# TODO implement hpc/slurm utility script (test-tube)
# TODO dupes autorunner via hpc/slurm utility^
