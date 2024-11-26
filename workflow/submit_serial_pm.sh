#!/bin/bash
#SBATCH -A m2616_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH --exclusive

module load pytorch
which python
export MPICH_GPU_SUPPORT_ENABLED=0

seed=20030
work_dir="/pscratch/sd/t/tianle/myWork/exalearn/ALBAND/"
exe_dir="${work_dir}/executable/"
exp_dir="${work_dir}/experiment/seed_${seed}/"
shared_file_dir="${exp_dir}/sfd/"
data_dir="${work_dir}/data/seed_${seed}/"
num_sample=4500
num_sample_val=$((${num_sample} / 2))
num_sample_test=$((${num_sample} / 2))
num_sample_study=${num_sample}
num_al_sample=$((${num_sample} * 3))
batch_size=512
epochs_0=400
epochs_1=300
epochs_2=250
epochs_3=200

NNODES=$SLURM_JOB_NUM_NODES

nthread=32
nthread_tot=$(( ${NNODES} * ${nthread} ))

nthread_study=22
nthread_study_tot=$(( ${NNODES} * ${nthread_study} ))

nrank_ml=4
nrank_ml_tot=$(( ${NNODES} * ${nrank_ml} ))

ngpus=$((${NNODES} * 4))
echo "Logging: Start! seed = ${seed}"
echo "Logging: data_dir = ${data_dir}"
echo "Logging: Doing cleaning"
rm -r ${exp_dir}
rm -r ${data_dir}

mkdir -p ${exp_dir}
cd ${exp_dir}

################################  Start real job  ##############################

{
    set -e
    very_start=$(date +%s%3N)

    python3 ${work_dir}/prepare_data_dir_pm.py --seed ${seed}
    
    echo "Logging: Start base simulation and merge!"
    start=$(date +%s%3N)
    srun --nodes ${NNODES} --ntasks ${nthread_tot} --cpus-per-task 2 --gpus=0 \
        python ${exe_dir}/simulation_sample.py \
               ${num_sample} ${seed} \
               ${data_dir}/base/config/config_1001460_cubic.txt \
               ${data_dir}/base/config/config_1522004_trigonal.txt \
               ${data_dir}/base/config/config_1531431_tetragonal.txt

    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data cubic ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data trigonal ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/base/data tetragonal ${nthread_tot}
    echo "Logging: End base simulation and merge, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start val simulation and merge!"
    start=$(date +%s%3N)
    srun --nodes ${NNODES} --ntasks ${nthread_tot} --cpus-per-task 2 --gpus=0 \
        python ${exe_dir}/simulation_sample.py \
               ${num_sample_val} $((${seed} - 1)) \
               ${data_dir}/validation/config/config_1001460_cubic.txt \
               ${data_dir}/validation/config/config_1522004_trigonal.txt \
               ${data_dir}/validation/config/config_1531431_tetragonal.txt

    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/validation/data cubic ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/validation/data trigonal ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/validation/data tetragonal ${nthread_tot}
    echo "Logging: End val simulation and merge, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start test simulation and merge!"
    start=$(date +%s%3N)
    srun --nodes ${NNODES} --ntasks ${nthread_tot} --cpus-per-task 2 --gpus=0\
        python ${exe_dir}/simulation_sample.py \
               ${num_sample_test} $((${seed} + 1)) \
               ${data_dir}/test/config/config_1001460_cubic.txt \
               ${data_dir}/test/config/config_1522004_trigonal.txt \
               ${data_dir}/test/config/config_1531431_tetragonal.txt

    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data cubic ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data trigonal ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/test/data tetragonal ${nthread_tot}
    echo "Logging: End test simulation and merge, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start study simulation and merge!"
    start=$(date +%s%3N)
    srun --nodes ${NNODES} --ntasks ${nthread_study_tot} --cpus-per-task 2 --gpus=0\
        python ${exe_dir}/simulation_sweep.py \
                ${num_sample_study} \
                ${data_dir}/study/config/config_1001460_cubic.txt \
                ${data_dir}/study/config/config_1522004_trigonal.txt \
                ${data_dir}/study/config/config_1531431_tetragonal.txt
 
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data cubic ${nthread_study_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data trigonal ${nthread_study_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/study/data tetragonal ${nthread_study_tot}
    echo "Logging: End study simulation and merge, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start training, phase 0"
    if [ -d ${shared_file_dir} ]; then
        rm -r ${shared_file_dir}
    fi
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    srun --nodes ${NNODES} --ntasks=${nrank_ml_tot} --cpus-per-task 32 --gpus-per-task=1 --gpu-bind=none\
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs_0} \
                                   --seed ${seed} \
                                   --device=gpu \
                                   --num_threads 32 \
                                   --phase_idx 0 \
                                   --data_dir ${data_dir} \
                                   --do_preprocess_study \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training phase 0, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start AL, phase 0"
    start=$(date +%s%3N)
    python ${exe_dir}/active_learning.py --seed $((${seed} + 1)) --num_new_sample ${num_al_sample} --policy uncertainty
    echo "Logging: End AL phase 0, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start resample simulation and merge, phase 1!"
    start=$(date +%s%3N)
    srun --nodes ${NNODES} --ntasks ${nthread_tot} --cpus-per-task 2 --gpus=0 \
        python ${exe_dir}/simulation_resample.py \
                $((${seed} + 2)) \
                ${data_dir}/AL_phase_1/config/config_1001460_cubic.txt \
                ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
                ${data_dir}/AL_phase_1/config/config_1522004_trigonal.txt \
                ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
                ${data_dir}/AL_phase_1/config/config_1531431_tetragonal.txt \
                ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5

    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data cubic ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data trigonal ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_1/data tetragonal ${nthread_tot}
    echo "Logging: End resample simulation and merge, phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start training, phase 1"
    if [ -d ${shared_file_dir} ]; then
        rm -r ${shared_file_dir}
    fi
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    srun --nodes ${NNODES} --ntasks=${nrank_ml_tot} --cpus-per-task 32 --gpus-per-task=1 --gpu-bind=none\
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs_1} \
                                   --seed ${seed} \
                                   --device=gpu \
                                   --num_threads 32 \
                                   --phase_idx 1 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training, phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    rm AL-freq.npy
 
    echo "Logging: Start AL, phase 1"
    start=$(date +%s%3N)
    python ${exe_dir}/active_learning.py --seed $((${seed} + 2)) --num_new_sample ${num_al_sample} --policy uncertainty
    echo "Logging: End AL phase 1, $(( $(date +%s%3N) - ${start} )) milliseconds"
 
    echo "Logging: Start resample simulation and merge, phase 2!"
    start=$(date +%s%3N)
    srun --nodes ${NNODES} --ntasks ${nthread_tot} --cpus-per-task 2 --gpus=0\
        python ${exe_dir}/simulation_resample.py \
                $((${seed} + 3)) \
                ${data_dir}/AL_phase_2/config/config_1001460_cubic.txt \
                ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
                ${data_dir}/AL_phase_2/config/config_1522004_trigonal.txt \
                ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
                ${data_dir}/AL_phase_2/config/config_1531431_tetragonal.txt \
                ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5
 
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data cubic ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data trigonal ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_2/data tetragonal ${nthread_tot}
    echo "Logging: End resample simulation and merge, phase 2, $(( $(date +%s%3N) - ${start} )) milliseconds"
    
    echo "Logging: Start training, phase 2"
    if [ -d ${shared_file_dir} ]; then
        rm -r ${shared_file_dir}
    fi
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    srun --nodes ${NNODES} --ntasks=${nrank_ml_tot} --cpus-per-task 32 --gpus-per-task=1 --gpu-bind=none\
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs_2} \
                                   --seed ${seed} \
                                   --device=gpu \
                                   --num_threads 32 \
                                   --phase_idx 2 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training, phase 2, $(( $(date +%s%3N) - ${start} )) milliseconds"
    
    rm AL-freq.npy

    echo "Logging: Start AL, phase 2"
    start=$(date +%s%3N)
    python ${exe_dir}/active_learning.py --seed $((${seed} + 3)) --num_new_sample ${num_al_sample} --policy uncertainty
    echo "Logging: End AL phase 2, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start resample simulation and merge, phase 3!"
    start=$(date +%s%3N)
    srun --nodes ${NNODES} --ntasks ${nthread_tot} --cpus-per-task 2 --gpus=0\
        python ${exe_dir}/simulation_resample.py \
               $((${seed} + 4)) \
               ${data_dir}/AL_phase_3/config/config_1001460_cubic.txt \
               ${data_dir}/study/data/cubic_1001460_cubic.hdf5 \
               ${data_dir}/AL_phase_3/config/config_1522004_trigonal.txt \
               ${data_dir}/study/data/trigonal_1522004_trigonal.hdf5 \
               ${data_dir}/AL_phase_3/config/config_1531431_tetragonal.txt \
               ${data_dir}/study/data/tetragonal_1531431_tetragonal.hdf5

    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data cubic ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data trigonal ${nthread_tot}
    python ${exe_dir}/merge_preprocess_hdf5.py ${data_dir}/AL_phase_3/data tetragonal ${nthread_tot}
    echo "Logging: End resample simulation and merge, phase 3, $(( $(date +%s%3N) - ${start} )) milliseconds"

    echo "Logging: Start training, phase 3"
    if [ -d ${shared_file_dir} ]; then
        rm -r ${shared_file_dir}
    fi
    mkdir -p ${shared_file_dir}
    start=$(date +%s%3N)
    srun --nodes ${NNODES} --ntasks=${nrank_ml_tot} --cpus-per-task 32 --gpus-per-task=1 --gpu-bind=none\
        python ${exe_dir}/train.py --batch_size ${batch_size} \
                                   --epochs ${epochs_3} \
                                   --seed ${seed} \
                                   --device=gpu \
                                   --num_threads 32 \
                                   --phase_idx 3 \
                                   --data_dir ${data_dir} \
                                   --shared_file_dir ${shared_file_dir}
    echo "Logging: End training, phase 3, $(( $(date +%s%3N) - ${start} )) milliseconds"
   
    rm AL-freq.npy
 
    echo "Logging: All done for seed = ${seed}"
    echo "Logging: End entire script, total takes $(( $(date +%s%3N) - ${very_start} )) milliseconds"
}
