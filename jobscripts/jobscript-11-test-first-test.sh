#!/bin/bash --login
### Choose ONE of the following partitions depending on your permitted access

#SBATCH -p gpuL              
### Required flags
#SBATCH -G 1                 # (or --gpus=N) Number of GPUs 
#SBATCH -t 4-0               # Wallclock timelimit (1-0 is one day, 4-0 is max permitted)

### Optional flags
#SBATCH -n 8          # (or --ntasks=) Number of CPU (host) cores (default is 1)
                             # See above for number of cores per GPU you can request.
                             # Also affects host RAM allocated to job unless --mem=num used.

#SBATCH --mail-type=ALL
#SBATCH --mail-user=chun.tham@student.manchester.ac.uk

module purge
conda deactivate

module load libs/cuda
conda activate nlu-proj-348

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

python3 -c "import torch;
print(f'torch cuda is available: {torch.cuda.is_available()}')"

python code/test_av-1.py --test_csv trial_data/AV_trial.csv --model_name microsoft/deberta-v3-xsmall --model_path checkpoints/av_siamese/deberta-v3-xsmall_ep2_bs512_lr0.002_ml2_m1.0_dl0_model.pt --batch_size 64 --max_len 512 --margin 1 --dist_lower 0 --output_path test/AV_trial_preds.csv

module purge
conda deactivate
