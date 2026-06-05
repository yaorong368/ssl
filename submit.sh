#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=32g
#SBATCH --gres=gpu:1
#SBATCH -p qTRDGPU
#SBATCH -t 48:00:00
#SBATCH -J vicreg_linear
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

# python linear_eval.py --data_dir ../datas/cifar10/ --ckpt ./runs/cifar10/zddp8/vicreg_ddp8_bs1024_dim1024_ep400/ckpt_latest.pt \
#     --img_size 32 --epochs 100 --batch_size 512 --lr 0.1 --num_workers 8 --amp --out_dir ./runs/cifar10/zddp8/vicreg_ddp8_bs1024_dim1024_ep400/linear_eval \
#     --method vicreg


# python linear_eval.py --data_dir ../datas/cifar10/ --ckpt ./runs/cifar10/byol_ddp1_bs2048_dim256_ep400/ckpt_latest.pt \
#     --img_size 32 --epochs 100 --batch_size 512 --lr 0.1 --num_workers 8 --amp --out_dir ./runs/cifar10/byol_ddp1_bs2048_dim256_ep400/linear_eval \
#     --method byol

python linear_eval.py --data_dir ../datas/imagenet100/ --ckpt ./runs/imagenet100/vicreg_ddp8_bs1024_dim1024_ep500/ckpt_latest.pt \
    --img_size 224 --epochs 100 --batch_size 128 --lr 0.1 --num_workers 4 --amp --out_dir ./runs/imagenet100/vicreg_ddp8_bs1024_dim1024_ep500/linear_eval \
    --method vicreg --image_net

# python linear_eval.py --data_dir ../datas/imagenet100/ --ckpt ./runs/imagenet100/byol_ddp8_bs1024_dim256_ep500/ckpt_latest.pt \
#     --img_size 224 --epochs 100 --batch_size 128 --lr 0.1 --num_workers 4 --amp --out_dir ./runs/imagenet100/byol_ddp8_bs1024_dim256_ep500/linear_eval \
#     --method byol --byol_pred_hidden_dim 2048 --image_net

# python main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/testtime/ --batch_size 128 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 50 --method jdrx \
#   --num_subsets 16

# python main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 2048 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-128 --epochs 400 --method simclr \
#   --num_subsets 32

# python main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/testtime/ --batch_size 512 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 50 --method jdrx \
#   --num_subsets 64

# python main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 2048 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method jdrx \
#   --num_subsets 256



# torchrun --standalone --nproc_per_node=1 linear_eval_ddp.py --data_dir ../datas/imagenet100/ --ckpt ./runs/imagenet100/simclr_ddp8_bs1024_dim1024_ep500/ckpt_latest.pt \
#     --img_size 224 --epochs 100 --batch_size 128 --lr 0.1 --num_workers 4 --amp --out_dir ./runs/imagenet100/test_ddp_linear/linear_eval_290 \
#     --method simclr --image_net

# torchrun --standalone --nproc_per_node=2 linear_eval_ddp.py --data_dir ../datas/imagenet100/ --ckpt ./runs/imagenet100/simclr_ddp8_bs1024_dim1024_ep500/ckpt_latest.pt \
#     --img_size 224 --epochs 100 --batch_size 128 --lr 0.1 --num_workers 4 --amp --out_dir ./runs/imagenet100/test_ddp_linear/linear_eval \
#     --method simclr --image_net

# python linear_eval.py --data_dir ../datas/stl10/ --ckpt ./runs/stl10/jdrx_ddp8_bs2048_dim1024_ep400_subsize32/ckpt_latest.pt \
#     --img_size 32 --epochs 100 --batch_size 512 --lr 0.1 --num_workers 8 --amp --out_dir ./runs/stl10/jdrx_ddp8_bs2048_dim1024_ep400_subsize32/linear_eval \
#     --method jdrx

# python linear_eval.py --data_dir ../datas/cifar10/ --ckpt ./runs/cifar10/simclr_ddp1_bs2048_dim1024_ep400/ckpt_latest.pt \
#     --img_size 32 --epochs 100 --batch_size 512 --lr 0.1 --num_workers 8 --amp --out_dir ./runs/cifar10/simclr_ddp1_bs2048_dim1024_ep400/linear_eval \
#     --method simclr


# python linear_eval.py --data_dir ../datas/cifar100/ --ckpt ./runs/cifar100/jdrx_ddp8_bs1024_dim1024_ep400_subsize8/ckpt_latest.pt \
#     --img_size 32 --epochs 100 --batch_size 512 --lr 0.1 --num_workers 4 --amp --out_dir ./runs/cifar100/jdrx_ddp8_bs1024_dim1024_ep400_subsize8/linear_eval \
#     --method jdrx


# python linear_eval.py --data_dir ../datas/cifar10/ --ckpt ./runs/cifar10/barlow_ddp8_bs1024_dim1024_ep400/ckpt_latest.pt \
#     --img_size 32 --epochs 100 --batch_size 512 --lr 0.1 --num_workers 8 --amp --out_dir ./runs/cifar10/barlow_ddp8_bs1024_dim1024_ep400/linear_eval \
#     --method barlow
