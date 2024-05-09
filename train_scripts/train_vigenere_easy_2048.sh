#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
singularity exec --nv -B /data:/data -B /data:/scratch/data /data/cs3450/pytorch20.11.3.sif bash -c '


echo "pip install done"
python /home/paganinik/vignereCracker/train_lstm_same_key.py \
--epochs 150 \
--hidden_size 2048 \
--cipher V \
--key AB \
--model_name vigenere_easy_key_2048 \
--log_name vigenere_easy_key_2048 \
--train_dataset_path data/amazon_train.txt
'