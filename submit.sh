#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=32g
#SBATCH --gres=gpu:A100:1
#SBATCH -p qTRDGPUM
#SBATCH -t 48:00:00
#SBATCH -J barlow_img
#SBATCH -A trends108c146
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yxiao11@student.gsu.edu
#SBATCH --output=/data/users2/yxiao11/model/ssl/%x_%j.out
#SBATCH --error=/data/users2/yxiao11/model/ssl/%x_%j.err

# Optional: load your CUDA module and activate conda environment

cd /data/users2/yxiao11/
source .bashrc
source activate p37

cd /data/users2/yxiao11/model/ssl

# # Make sure the GPUs are visible
# export CUDA_VISIBLE_DEVICES=0,1,2,3


# Run the training script

python main.py --data_dir ../datas/imagenet100/ --out_dir ./runs/imagenet100/ --batch_size 256 --img_size 32 --amp --num_workers 4 --projector 512-512 --epochs 1000 --method barlow
# python main.py --data_dir ../datas/imagenet100/ --out_dir ./runs/imagenet100/ --batch_size 256 --img_size 32 --amp --num_workers 4 --projector 512-512 --epochs 1000 --method jdrx --num_subsets 8