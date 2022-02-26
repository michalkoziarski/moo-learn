#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -p plgrid-gpu
#SBATCH --time=72:00:00
#SBATCH --gres=gpu
#SBATCH -A plgimba2

module add plgrid/tools/python/3.9

python3 -W ignore ${1} ${@:2}
