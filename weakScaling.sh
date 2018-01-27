#!/bin/bash
#PBS -l nodes=3:ppn=8,walltime=00:30:00
#PBS -N weakScaling


 
cd $PBS_O_WORKDIR
export TIMEFORMAT=%R
export OMP_NUM_THREADS=1

module purge
module load extras Xlibraries emacs git
module load gcc/4.8.1
module load openmpi/gcc/1.6.4

for nodes in 1 2 4 8 16 24
do
{ time mpirun -np $nodes ./diffuse diffuseparams$nodes.txt ;} 2>> time-weakScaling.txt
done
