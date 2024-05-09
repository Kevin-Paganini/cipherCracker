#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
singularity exec --nv -B /data:/data -B /data:/scratch/data /data/cs3450/pytorch20.11.3.sif bash -c '


echo "pip install done"
python /home/paganinik/vignereCracker/train_lstm_same_key.py \
--epochs 250 \
--hidden_size 2048 \
--cipher C \
--key 4 \
--model_name ceaser_cipher_2048_lite \
--log_name ceaser_cipher_2048_lite \
--train_dataset_path data/amazon_train_lite.txt
'