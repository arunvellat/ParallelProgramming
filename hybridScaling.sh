#!/bin/bash
#PBS -l nodes=3:ppn=8,walltime=03:00:00
#PBS -N hybridScaling

cd $PBS_O_WORKDIR
TIMEFORMAT=%R
maxcores=24

module purge
module load extras Xlibraries emacs git
module load gcc/4.8.1
module load openmpi/gcc/1.6.4

for thread  in {1,2,4,8}
do
export OMP_NUM_THREADS=$thread
np=$(expr $maxcores / $thread)
file=thread-$thread.txt

for process in {1,2,4,8}
do
 if [ $(expr  $process \* $thread) -le $maxcores ]
     then
      { time mpirun -np $process --bynode ./diffuse diffuseparams.txt ;} 2>> $file 
       
     
    
 fi
done

{ time mpirun -np $np --bynode ./diffuse diffuseparams.txt ;} 2>> $file ;

done
