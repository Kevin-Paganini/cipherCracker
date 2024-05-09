#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
singularity exec --nv -B /data:/data -B /data:/scratch/data /data/cs3450/pytorch20.11.3.sif bash -c '


echo "pip install done"
python /home/paganinik/vignereCracker/train_lstm_rand_key.py \
--epochs 500 \
--hidden_size 2048 \
--cipher A \
--key DWITVAYNE \
--model_name autokey_cipher_2048_rand \
--log_name autokey_cipher_2048_rand \
'