#!/bin/bash
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
singularity exec --nv -B /data:/data -B /data:/scratch/data /data/cs3450/pytorch20.11.3.sif bash -c '


echo "pip install done"
python /home/paganinik/vignereCracker/train_lstm_same_key.py \
--epochs 100 \
--hidden_size 1024 \
--cipher V \
--key RWPFUQBZIRASDLJFUTBSMTRYS \
--model_name vigenere_hard_cipher_1024 \
--log_name vigenere_hard_cipher_1024 \
'