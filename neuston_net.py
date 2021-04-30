#!/usr/bin/env python
"""the main thing"""

# built in imports
from shutil import copyfile
import argparse
import os, glob
import datetime as dt

# 3rd party imports
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger,ExperimentWriter
from torchvision.datasets.folder import IMG_EXTENSIONS

# project imports
import ifcb
from neuston_models import NeustonModel
from neuston_callbacks import SaveValidationResults, SaveTestResults
from neuston_data import get_trainval_datasets, IfcbBinDataset, ImageDataset

## NOTES ##
# https://pytorch-lightning.readthedocs.io/en/0.8.5/introduction_guide.html


def main(args):

    if args.cmd_mode=='TRAIN':
        do_training(args)
    else: # RUN
        do_run(args)

    print('\nDONE!')


def do_training(args):

    # ARG CORRECTIONS AND CHECKS
    date_str = args.cmd_timestamp.split('T')[0]
    args.model_id = args.model_id.format(TRAIN_DATE=date_str, TRAIN_ID=args.TRAIN_ID)

    # make sure output directory exists
    os.makedirs(args.outdir,exist_ok=True)

    # Setup Callbacks
    callbacks=[]
    plotting_callbacks = [] # TODO

    validation_results_callbacks = []
    if not args.result_files:
        args.result_files = ['results.mat training_image_basenames training_classes image_basenames input_classes output_scores confusion_matrix counts_perclass f1_perclass f1_weighted f1_macro'.split()]
    for result_file in args.result_files:
        svr = SaveValidationResults(outdir=args.outdir, outfile=result_file[0], series=result_file[1:])
        validation_results_callbacks.append(svr)
    callbacks.extend(validation_results_callbacks)
    callbacks.extend(plotting_callbacks)

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

    # Gerry Rig Logger
    class ExperimentWriter_hack(ExperimentWriter):
        def log_metrics(self, metrics_dict, step=None):
            _handle_value = lambda v: v.item() if isinstance(v, torch.Tensor) else v
            metrics = {k: _handle_value(v) for k, v in metrics_dict.items()
                       if k not in ['input_classes', 'output_classes', 'input_srcs', 'outputs']}
            self.metrics.append(metrics)
    logger = CSVLogger(save_dir=os.path.join(args.outdir,'logs'), name='default', version=None)
    os.makedirs(logger.root_dir, exist_ok=True)
    logger._experiment = ExperimentWriter_hack(log_dir=logger.log_dir)

    # Setup Trainer
    chkpt_path = os.path.join(args.outdir, 'chkpts')
    os.makedirs(chkpt_path, exist_ok=True)
    trainer = Trainer(deterministic=True, logger=logger,
                      gpus=len(args.gpus) if args.gpus else None,
                      max_epochs=args.emax, min_epochs=args.emin,
                      early_stop_callback=EarlyStopping('val_loss',patience=args.estop) if args.estop else False,
                      checkpoint_callback=ModelCheckpoint(filepath=chkpt_path),
                      callbacks=callbacks,
                      num_sanity_val_steps=0
                      )

    # Setup Model
    classifier = NeustonModel(args)
    # TODO setup dataloaders in the model, allowing auto-batch-size optimization
    # see https://pytorch-lightning.readthedocs.io/en/stable/training_tricks.html#auto-scaling-of-batch-size

    # Do Training
    trainer.fit(classifier, train_dataloader=training_loader, val_dataloaders=validation_loader)

    # Copy best model
    checkpoint_path = trainer.checkpoint_callback.best_model_path
    output_path = os.path.join(args.outdir, args.model_id+'.ptl')
    copyfile(checkpoint_path, output_path)

    # Copying Logs
    if args.epochs_log:
        output_path = os.path.join(args.outdir, args.epochs_log)
        copyfile(logger.experiment.metrics_file_path, output_path)
    if args.args_log:
        src_path = os.path.join(logger.experiment.log_dir, logger.experiment.NAME_HPARAMS_FILE)
        output_path = os.path.join(args.outdir, args.args_log)
        copyfile(src_path, output_path)


def do_run(args):

    # assert correct filter arguments
    if args.filter:
        if not args.filter[0] in ['IN', 'OUT']:
            argparse.ArgumentTypeError('IN|OUT must be either "IN" or "OUT"')
        if len(args.filter) < 2:
            argparse.ArgumentTypeError('Must be at least one KEYWORD')

    # load model
    classifier = NeustonModel.load_from_checkpoint(args.MODEL)
    seed_everything(classifier.hparams.seed)

    # ARG CORRECTIONS AND CHECKS
    if os.path.isdir(args.SRC) and not args.SRC.endswith(os.sep): args.SRC = args.SRC+os.sep

    # set OUTFILE defaults
    if not args.outfile:
        if args.src_type == 'bin': args.outfile=['D{BIN_YEAR}/D{BIN_DATE}/{BIN_ID}_class.h5']
        if args.src_type == 'img': args.outfile = ['img_results.json']

    # Setup Callbacks
    plotting_callbacks = []  # TODO
    run_results_callbacks = []
    for outfile in args.outfile:
        svr = SaveTestResults(outdir=args.outdir, outfile=outfile, timestamp=args.cmd_timestamp)
        run_results_callbacks.append(svr)

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
    image_loaders = []
    if args.src_type == 'bin':
        # Formatting Dataset
        if os.path.isdir(args.SRC):
            if filter_mode=='IN':
                dd = ifcb.DataDirectory(args.SRC, whitelist=filter_keywords)
            elif filter_mode=='OUT':
                dd = ifcb.DataDirectory(args.SRC, blacklist=filter_keywords)
            else:
                dd = ifcb.DataDirectory(args.SRC)
        elif os.path.isfile(args.SRC) and args.SRC.endswith('.txt'): # TODO TEST: textfile bin run
            with open(args.SRC,'r') as f:
                bins = f.readlines()
            parent = os.path.commonpath(bins)
            dd = ifcb.DataDirectory(parent,whitelist=bins)
        else: # single bin # TODO TEST: single bin run
            parent = os.path.dirname(args.SRC)
            bin_id = os.path.basename(args.SRC)
            dd = ifcb.DataDirectory(parent,whitelist=[bin_id])

        error_bins = []

        if args.gobig: print('Loading Bins',end=' ')
        for i, bin_fileset in enumerate(dd):
            bin_fileset.pid.namespace = os.path.dirname(bin_fileset.fileset.basepath.replace(args.SRC,''))+os.sep
            bin_obj = bin_fileset.pid
            if args.filter: # applying filter
                if filter_mode=='IN': # if bin does NOT match any of the keywords, skip it
                    if not any([k in str(bin_obj) for k in filter_keywords]): continue
                elif filter_mode=='OUT': # if bin matches any of the keywords, skip it
                    if any([k in str(bin_obj) for k in filter_keywords]): continue

            if not args.clobber:
                output_files = [os.path.join(args.outdir, ofile) for ofile in args.outfile]
                outfile_dict = dict(BIN_ID=bin_obj.pid,
                                    BIN_YEAR=bin_obj.year,
                                    BIN_DATE=bin_obj.yearday,
                                    INPUT_SUBDIRS=bin_obj.namespace)
                output_files = [ofile.format(**outfile_dict).replace(2*os.sep,os.sep) for ofile in output_files]
                if all([ os.path.isfile(ofile) for ofile in output_files ]):
                    print('{} result-file(s) already exist - skipping this bin'.format(bin_obj))
                    continue

            bin_dataset = IfcbBinDataset(bin_fileset, classifier.hparams.resize, classifier.hparams.img_norm)
            image_loader = DataLoader(bin_dataset, batch_size=args.batch_size,
                                      pin_memory=True, num_workers=args.loaders)

            # skip empty bins
            if len(image_loader) == 0:
                error_bins.append((bin_obj, AssertionError('Bin is Empty')))
                continue
            if args.gobig:
                print('.',end='',flush=True)
                image_loaders.append(image_loader)
            else:
                # Do runs one bin at a time
                try: trainer.test(classifier, test_dataloaders=image_loader)
                except Exception as e:
                    error_bins.append((bin_obj,e))

        # Do Runs all at once
        if args.gobig: print(); trainer.test(classifier, test_dataloaders=image_loaders)

        # Final Statements
        print('RUN IS DONE')
        if error_bins:
            print("The following bins failed; they were not processed:")
            for bin_obj,err in error_bins:
                print(bin_obj,type(err),err)

    ## IMAGES ##
    else:
        img_paths = []
        if os.path.isdir(args.SRC):
            for img in glob.iglob(os.path.join(args.SRC,'**','**'), recursive=True):
                if img.endswith(IMG_EXTENSIONS):
                    img_paths.append(img)
        elif os.path.isfile(args.SRC) and args.SRC.endswith('.txt'): # TODO TEST: textfile img run
            with open(args.SRC,'r') as f:
                img_paths = f.readlines()
                img_paths = [img.strip() for img in img_paths]
                img_paths = [img for img in img_paths if img.endswith(IMG_EXTENSIONS)]
        elif args.SRC.endswith(IMG_EXTENSIONS): # single img # TODO TEST: single img run
            img_paths.append(args.SRC)

        # applying filter
        if args.filter:
            for img in img_paths[:]:
                if filter_mode=='IN': # if img does NOT match any of the keywords, skip it
                    if not any([k in img for k in filter_keywords]): img_paths.remove(img)
                elif filter_mode=='OUT': # if img matches any of the keywords, skip it
                    if any([k in img for k in filter_keywords]): img_paths.remove(img)

        assert len(img_paths)>0, 'No images to process'
        image_dataset = ImageDataset(img_paths, resize=classifier.hparams.resize, input_src=args.SRC)
        image_loader = DataLoader(image_dataset, batch_size=args.batch_size,
                                  pin_memory=True, num_workers=args.loaders)

        trainer.test(classifier,test_dataloaders=image_loader)


def argparse_nn(parser=None):

    if parser is None:
        parser = argparse.ArgumentParser(description='Train, Run, and perform other tasks related to ifcb and general image classification!')
        # TODO move most of these parser hparams to respective pytorch-lightning objects?

    # Create subparsers
    subparsers = parser.add_subparsers(dest='cmd_mode', help='These sub-commands are mutually exclusive. Note: optional arguments (below) must be specified before "TRAIN" or "RUN"')
    train = subparsers.add_parser('TRAIN', help='Train a new model')
    run = subparsers.add_parser('RUN', help='Run a previously trained model')

    ## Common Vars ##
    common = parser.add_argument_group(title='NN Common Args', description=None)
    common.add_argument('--batch', dest='batch_size', metavar='SIZE', default=108, type=int, help='Number of images per batch. Defaults is 108') # todo: auto-mode built in to ptl
    common.add_argument('--loaders', metavar='N', default=4, type=int, help='Number of data-loading threads. 4 per GPU is typical. Default is 4') # todo: auto-mode?

    argparse_nn_train(train)
    argparse_nn_run(run)

    return parser

def argparse_nn_train(train_subparser):
    ## Training Vars ##
    train_subparser.add_argument('SRC', help='Directory with class-label subfolders and images. May also be a dataset-configuration csv.')
    train_subparser.add_argument('MODEL', help='Select a base model. Eg: "inception_v3"')
    # TODO choices field. TODO: "Accepts a known model name, or a path to a specific model file for transfer learning"
    train_subparser.add_argument('TRAIN_ID', help='Training ID. This value is the default value used by --outdir and --model-id.')

    model = train_subparser.add_argument_group(title='Model Adjustments', description=None)
    model.add_argument('--untrain', dest='pretrained', default=True, action='store_false',
                       help='If set, initializes MODEL ~without~ pretrained neurons. Default (unset) is pretrained')
    model.add_argument('--img-norm', nargs=2, metavar=('MEAN', 'STD'),
                       help='Normalize images by MEAN and STD. This is like whitebalancing. '
                            'eg1: "0.667 0.161", eg2: "0.056,0.058,0.051 0.067,0.071,0.057"')
    # TODO layer freezing and transfer learning params.

    data = train_subparser.add_argument_group(title='Dataset Adjustments', description=None)
    data.add_argument('--seed', default=0, type=int, help='Set a specific seed for deterministic output & dataset-splitting reproducability.')
    data.add_argument('--split', metavar='T:V', default='80:20', help='Ratio of images per-class to split randomly into Training and Validation datasets. Randomness affected by SEED. Default is "80:20"')
    data.add_argument('--class-config', metavar=('CSV', 'COL'), nargs=2, help='Skip and combine classes as defined by column COL of a special CSV configuration file')
    data.add_argument('--class-min', metavar='MIN', default=2, type=int, help='Exclude classes with fewer than MIN instances. Default is 2')
    data.add_argument('--class-max', metavar='MAX', default=None, type=int, help='Limit classes to a MAX number of instances. '
                           'If multiple datasets are specified with a dataset-configuration csv, classes from lower-priority datasets are truncated first.')
    data.add_argument('--swap', default=False, action='store_true',
                      help=argparse.SUPPRESS)  # dupes placeholder. may not be needed.

    epochs = train_subparser.add_argument_group(title='Epoch Parameters', description=None)
    epochs.add_argument('--emax', metavar='MAX', default=60, type=int, help='Maximum number of training epochs. Default is 60')
    epochs.add_argument('--emin', metavar='MIN', default=10, type=int, help='Minimum number of training epochs. Default is 10')
    epochs.add_argument('--estop', metavar='STOP', default=10, type=int, help='Early Stopping: Number of epochs following a best-epoch after-which to stop training. Set STOP=0 to disable. Default is 10')

    augs = train_subparser.add_argument_group(title='Augmentation Options', description='Data Augmentation is a technique by which training results may improved by simulating novel input')
    augs.add_argument('--flip', choices=['x', 'y', 'xy', 'x+V', 'y+V', 'xy+V'],
                      help='Training images have 50%% chance of being flipped along the designated axis: (x) vertically, (y) horizontally, (xy) either/both. May optionally specify "+V" to include Validation dataset')

    out = train_subparser.add_argument_group(title='Output Options')
    out.add_argument('--outdir', default='training-output/{TRAIN_ID}', help='Default is "training-output/{TRAIN_ID}"')
    out.add_argument('--model-id', default='{TRAIN_ID}', help='Set a specific model id. Patterns {TRAIN_DATE} and {TRAIN_ID} are recognized. Default is "{TRAIN_ID}"')
    out.add_argument('--epochs-log', metavar='ELOG', default='epochs.csv', help='Specify a csv filename. Includes epoch, loss, validation loss, and f1 scores. Default is epochs.csv')
    out.add_argument('--args-log', metavar='ALOG', default='args.yml', help='Specify a human-readable yaml filename. Includes all user-specified and default training parameters. Default is args.yml')
    out.add_argument('--results', dest='result_files', metavar=('FNAME', 'SERIES'), nargs='+', action='append',
                     help='FNAME: Specify a validation-results filename or pattern. Valid patterns are: "{epoch}". Accepts .json .h5 and .mat file formats.'
                          'SERIES: Data to include in FNAME. The following are always included and need not be specified: model_id, timestamp, class_labels, input_classes, output_classes.'
                          '    Options are: image_basenames, image_fullpaths; output_scores, output_winscores; confusion_matrix;'
                          '                 classes_by_{count|f1|recall|precision}; {f1|recall|precision}_{macro|weighted|perclass}; {counts|val_counts|train_counts}_perclass.'
                          '--results may be specified multiple times in order to create different files. '
                          'If not invoked, default is "results.mat training_image_basenames training_classes image_basenames input_classes output_scores confusion_matrix counts_perclass f1_perclass f1_weighted f1_macro"')
    #out.add_argument('-p','--plot', metavar=('FNAME','PARAM'), nargs='+', action='append', help='Make Plots') # TODO plots

    meta = train_subparser.add_argument_group(title='Metadata and Annotations')
    meta.add_argument('--dataset-id', help='Associate a dataset id label with this model')
    meta.add_argument('--notes', help='Add any kind of note to the trained model. Make sure to use quotes "around your message."')

    #optim = train_subparser.add_argument_group(title='Optimization', description='Adjust learning hyper parameters')
    #optim.add_argument('--optimizer', default='Adam', choices=['Adam'], help='Select and optimizer. Default is Adam')
    #optim.add_argument('--learning-rate',default=0.001,type=float,help='Set a learning rate. Default is 0.001')
    #optim.add_argument('--weight-decay', default='?', help="not sure where this comes in")
    #optim.add_argument('--class-norm', help='Bias results to emphasize smaller classes')
    #optim.add_argument('--batch-norm', help='i forget what this is exactly')


def argparse_nn_run(run_subparser):
    ## Run Vars ##
    run_subparser.add_argument('SRC', help='Resource(s) to be classified. Accepts a bin, an image, a text-file, or a directory. Directories are accessed recursively')
    run_subparser.add_argument('MODEL', help='Path to a previously-trained model file')
    run_subparser.add_argument('RUN_ID', help='Run ID. Used by --outdir')

    run_subparser.add_argument('--type', dest='src_type', default='bin', choices=['bin','img'], help='File type to perform classification on. Defaults is "bin"')
    run_subparser.add_argument('--outdir', default='run-output/{RUN_ID}/v3/{MODEL_ID}', help='Default is "run-output/{RUN_ID}/v3/{MODEL_ID}"')
    run_subparser.add_argument('--outfile', action='append',
        help='''Name/pattern of the output classification file.
                If TYPE==bin, files are created on a per-bin basis. OUTFILE must include "{BIN_ID}", which will be replaced with the a bin's id.
                A few patters are recognized: {BIN_ID}, {BIN_YEAR}, {BIN_DATE}, {INPUT_SUBDIRS}.
                A few output file formats are recognized: .json, .mat, and .h5 (hdf).
                Default for TYPE==bin is "D{BIN_YEAR}/D{BIN_DATE}/{BIN_ID}_class.h5"; Default for TYPE==img is "img_results.json".
             ''')
    run_subparser.add_argument('--filter', nargs='+', metavar=('IN|OUT','KEYWORD'),
        help='Explicitly include (IN) or exclude (OUT) bins or image-files by KEYWORDs. KEYWORD may also be a text file containing KEYWORDs, line-deliminated.')
    run_subparser.add_argument('--clobber', action='store_true',
        help='If set, already processed bins in OUTDIR are reprocessed. By default, if an OUTFILE exists already the associated bin is not reprocessed.')
    run_subparser.add_argument('--gobig', action='store_true', help=argparse.SUPPRESS)  # aggregates bins
    #run_subparser.add_argument('-p','--plot', metavar=('FNAME','PARAM'), nargs='+', action='append', help='Make Plots') # TODO plots

def argparse_nn_runtimeparams(args):
    # add timestamp
    args.cmd_timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')

    # add version tag
    try:
        with open('version') as f:
            args.version = f.read().strip()
    except FileNotFoundError:
        args.version = None

    ## GPU torch setup ## - use all available GPU's
    # pytorch uses the ~index~ of CUDA_VISIBLE_DEVICES.
    # So if gpus == [3,4] then device "cuda:0" == GPU no. 3
    #                      and device "cuda:1" == GPU no. 4
    if torch.cuda.is_available():
        args.gpus = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else: args.gpus = None

    # parse args.outdir value
    proc_outdir(args)


def proc_outdir(args):
    run_date_str, run_time_str = args.cmd_timestamp.split('T')
    if args.cmd_mode=='TRAIN':
        args.outdir = args.outdir.format(TRAIN_DATE=run_date_str, TRAIN_ID=args.TRAIN_ID)
    elif args.cmd_mode=='RUN':
        model_id = NeustonModel.load_from_checkpoint(args.MODEL).hparams.model_id
        args.outdir = args.outdir.format(RUN_DATE=run_date_str, RUN_ID=args.RUN_ID, MODEL_ID=model_id)


if __name__ == '__main__':

    parser = argparse_nn()
    input_args = parser.parse_args()
    argparse_nn_runtimeparams(input_args)
    main(input_args)

# TODO move dataloaders to NeustonModel for auto-batch-size enabling
# TODO implement plots (matplotlib vs plotly?)
# TODO dupes autorunner via hpc/slurm utility^
# update conda env: conda env update -f environment.yml --prune
# Quick hpc access: ssh poseidon; ./gpu_ifcbnn.sh
# TODO unittests?
