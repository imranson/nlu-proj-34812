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

python code/test_av-1.py --test_csv training_data/AV/dev.csv --model_name microsoft/deberta-v3-xsmall --model_path checkpoints/av_siamese/siamese-final-1/deberta-v3-xsmall_ep300_bs16_lr1e-05_ml512_m0.5_dl0_model.pt --batch_size 128 --max_len 512 --margin 0.5 --dist_lower 0 --output_path training_data/AV/dev_preds.csv

module purge
conda deactivate
