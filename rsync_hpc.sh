#!/usr/bin/env bash

# This script rsync's top level files from local project
# to remote hpc, and queue's an sbatch job.

src="/home/sbatchelder/Documents/ifcb"
proj_name=$(basename "$src")
host="sbatchelder@poseidon.whoi.edu"
hpc_home="/vortexfs1/home/sbatchelder/$proj_name"
hpc_scratch="/vortexfs1/scratch/sbatchelder/$proj_name"


flags=(-ruptv --exclude='*).py' --include='*.py' --include='batches/' --include='*.sbatch' --exclude='*')
# r recursivly send files
# u don't overwrite newer files
# p preserve permissions
# t preserve modification time
# excludes everything EXCEPT
# includes python and sbatch files./r


echo ">> rsync ${flags[@]} $src/ $host:hpc_scratch"
rsync ${flags[@]} $src/ $host:$hpc_scratch
rsync ${flags[@]} $src/ $host:$hpc_home



