#!/bin/bash
# specify a partition
#SBATCH -p stampede
# specify number of nodes
#SBATCH -N 3
#SBATCH -w mscluster[31,32,33]
# specify number of tasks
#SBATCH --ntasks=3
# specify number of cores (or threads) per process
##SBATCH -c 8
# specify memory pool per node (for all cores on the node)
##SBATCH --mem 20
# specify the wall clock time limit for the job (hh:mm:ss) 5 mins
##SBATCH -t 5:00
# specify the job name
#SBATCH -J KNearest
# specify the filename to be used for writing output
# change the path for your own account here
#SBATCH -o /home-mscluster/anana/test/KNearest/output/%N.%j.out
# specify the filename for stderr
# change the path for your own account here
#SBATCH -e /home-mscluster/anana/test/KNearest/output/error/%N.%j.err
echo ------------------------------------------------------
echo DATE: $(date)
echo 'Job is running on nodes ' $SLURM_JOB_NODELIST
echo ------------------------------------------------------
echo SLURM: sbatch is running on $SLURM_SUBMIT_HOST
echo SLURM: job ID is $SLURM_JOB_ID
echo SLURM: submit directory is $SLURM_SUBMIT_DIR
echo SLURM: number of nodes allocated is $SLURM_JOB_NUM_NODES
echo SLURM: number of tasks is $SLURM_NTASKS
echo SLURM: job name is $SLURM_JOB_NAME
echo ------------------------------------------------------
cd $SLURM_SUBMIT_DIR

sizes=(10000 100000 1000000)
dimension=(64 128 256)

echo ---------Starting Benchmark------------
echo =======================================
for i in "${dimension[@]}"
do
    echo "Testing Dimension: ${i}"
    echo =======================================
    echo Serial Version
    make
    for size in "${sizes[@]}"
    do
        echo "Testing array size ${size}"
            srun ./KNearest ${size} ${i}
        echo ---------------------------------------
    done
    echo Parallel Tasks Version
    make task
    for size in "${sizes[@]}"
    do
        echo "Testing array size ${size}"
            srun ./KNearest ${size} ${i}
        echo ---------------------------------------
    done
    echo Parallel Section Version
    make section
    for size in "${sizes[@]}"
    do
        echo "Testing array size ${size}"
            srun ./KNearest ${size} ${i}
        echo ---------------------------------------
    done
    echo =======================================
done
echo --------------Clean Files--------------
make clean
echo ---------Benchmark Completed-----------
