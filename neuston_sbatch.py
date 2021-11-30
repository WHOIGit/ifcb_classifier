#!/usr/bin/env python

import os, argparse
import neuston_net as nn
import sys
import shutil
import subprocess

CUDA101_MODULES = 'cuda10.1/toolkit cuda10.1/blas cuda10.1/cudnn/8.0.2 cuda10.1/fft'
default_cwd = os.path.dirname(os.path.abspath( __file__ ))
default_email = '{}@whoi.edu'.format( os.getlogin() )

## SINGLE SBATCH TEMPLATE ##
SBATCH_TEMPLATE = """#!/bin/sh
#SBATCH --job-name={JOB_NAME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPU_NUM}
#SBATCH --mem-per-cpu={MEM_PER_CPU}
#SBATCH --time={WALLTIME}
#SBATCH --mail-type=ALL
#SBATCH --mail-user={EMAIL}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:{GPU_NUM}
#SBATCH --output={SLURM_LOG_DIR}/{SLURM_LOG_FILE}

# SETTING OPERATIVE DIRECTORY #
cd {ABS_CWD}

# LOGGING JOB DETAILS #
echo "Job ID: $SLURM_JOB_ID, JobName: $SLURM_JOB_NAME"
hostname; pwd; date

# SETTING UP ENVIRONMENT #
module load {CUDA_MODULES}
module load anaconda
source activate {CONDA_ENV}
echo "Environment... Loaded"

# DO COMMAND #
{CMD}

"""

## DEFAULT PARAMETERS ##
SBATCH_DDICT = dict(JOB_NAME='NN', EMAIL=default_email, WALLTIME='24:00:00',
                    CUDA_MODULES=CUDA101_MODULES, CONDA_ENV='ifcbnn',
                    GPU_NUM=1, CPU_NUM=4, MEM_PER_CPU=10240,  #10GB
                    SLURM_LOG_DIR='slurm-logs', SLURM_LOG_FILE='%j.%x.out',
                    ABS_CWD=default_cwd)


def main(parser):
    SBATCH_DICT = SBATCH_DDICT.copy()
    os.chdir(SBATCH_DICT['ABS_CWD'])

    args = parser.parse_args()
    if args.cmd_mode is None:
        parser.error('Positional Argument "TRAIN" or "RUN" must be specified.')
    nn.argparse_nn_runtimeparams(args)

    # if any slurm params are set by user, overwrite the default sbatch dict template value
    for key in SBATCH_DICT:
        arg = getattr(args,key.lower(),None)
        if arg is not None:
            SBATCH_DICT[key] = arg

    if args.slurm_log_dir is None and 'outdir' in args:
        SBATCH_DICT['SLURM_LOG_DIR'] = args.outdir

    os.makedirs(SBATCH_DICT['SLURM_LOG_DIR'], exist_ok=True)

    # fetch only the arguments destined for neuston_net.py by finding the TRAIN or RUN index from sys.argv
    idx = sys.argv.index(args.cmd_mode)
    nn_args = sys.argv[idx:]

    # quotes need to be added back where needed and args appended to nn command
    nn_args = [arg if ' ' not in arg else '"{}"'.format(arg) for arg in nn_args]
    SBATCH_DICT['CMD'] = cmd = '''python neuston_net.py {}'''.format(' '.join(nn_args))
    print('SRUN Command:  '+cmd)

    # creating sbatch file
    sbatch_content = SBATCH_TEMPLATE.format(**SBATCH_DICT)
    sbatch_ofile_dict = dict(OUTDIR=args.outdir, JOB_NAME=SBATCH_DICT['JOB_NAME'])
    if not args.dry_run:
        tmp_fname = '/tmp/neuston_tmp.sbatch'
        with open(tmp_fname,'w') as f:
            f.write(sbatch_content)
        resp = subprocess.run(['sbatch',tmp_fname], universal_newlines=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if resp.returncode == 0:
            pid = resp.stdout.split()[-1]
            sbatch_ofile_dict['PID'] = pid
            print('SLURM job_id:  '+pid)
        else:
            print('SLURM job_id:  '+resp.stderr.strip().replace('sbatch: error:','<sbatch error>'))
            sbatch_ofile_dict['PID'] = 'xxxxxx'
    else:
        sbatch_ofile_dict['PID'] = 'xxxxxx'

    # record sbatch file to outdir directory
    sbatch_ofile = args.ofile.format(**sbatch_ofile_dict)
    print('SBATCH script: ' + sbatch_ofile)
    with open(sbatch_ofile,'w') as f:
        f.write(sbatch_content)


def argparse_sbatch():
    parser = argparse.ArgumentParser(description='SLURM SBATCH auto-submitter for neuston_net.py')

    slurm = parser.add_argument_group(title='SLURM Args', description=None)
    slurm.add_argument('--job-name', metavar='STR',
        help='Job Name that will appear in slurm jobs list. Defaults is "{}"'.format(SBATCH_DDICT['JOB_NAME']))
    slurm.add_argument('--email',
        help='Email address to send slurm notifications to. Your default is "{}"'.format(SBATCH_DDICT['EMAIL']))
    slurm.add_argument('--walltime', metavar='HH:MM:SS',
                       help='Set Slurm Task max runtime. Default is "{}"'.format(SBATCH_DDICT['WALLTIME']))
    slurm.add_argument('--gpu-num', metavar='INT', type=int,
        help='Number of GPUs to allocate per task. Default is {}'.format(SBATCH_DDICT['GPU_NUM']))
    slurm.add_argument('--cpu-num', metavar='INT', type=int,
        help='Number of CPUs to allocate per task. Default is {}'.format(SBATCH_DDICT['CPU_NUM']))
    slurm.add_argument('--mem-per-cpu', metavar='MB', type=int,
        help='Memory to allocate per cpu in MB. Default is {}MB'.format(SBATCH_DDICT['MEM_PER_CPU']))
    slurm.add_argument('--slurm-log-dir', metavar='DIR',
        help='Directory to save slurm log file to. Defaults to OUTDIR (as defined by TRAIN or RUN subcommand)')
    slurm.add_argument('--ofile', default="{OUTDIR}/{PID}.{JOB_NAME}.sbatch",
        help='Save location for generated sbatch file. Defaults to "{OUTDIR}/{PID}.{JOB_NAME}.sbatch"')
    slurm.add_argument('--conda-env', default='ifcbnn', help='The conda environment to activate for neuston_net.py. Default is "ifcbnn"')
    slurm.add_argument('--dry-run', default=False, action='store_true', help='Create the sbatch script but do not run it')

    return parser

if __name__ == '__main__':
    parser = argparse_sbatch()

    parser = nn.argparse_nn(parser)

    main(parser)


#TODO ok this test_tube thing is stupid. 
