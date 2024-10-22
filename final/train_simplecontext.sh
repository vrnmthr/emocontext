#!/bin/bash
# Note that the lines with $ are the qsub flags
# Execute from the current working directory
#$ -cwd
#
#  This is a long-running job so add this flag
#  We want to use 1 gpu
#$ -l gpus=1
# Specify what folder to save outputs and errors
# Note that these directories can be the same
#$ -o output_folder/
#$ -e error_folder/
#
# If you want to get e-mail notification you can do the following
#$ -m abes
# Note that you do not need to add all the letters (abes) check link for detail

# activate you Virtual environment
source /course/cs2952d/pytorch-gpu/bin/activate

# Run your command
python3 simple_context.py ../data/emocontext --device cuda --message temp-run-small-data
