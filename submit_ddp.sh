#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=250g
#SBATCH --gres=gpu:A100:8
#SBATCH -p qTRDGPUH
#SBATCH --time=5-00:00:00
#SBATCH -J jdrx_im
#SBATCH -A trends108c146
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yxiao11@student.gsu.edu
#SBATCH --output=/data/users2/yxiao11/model/ssl/%x_%j.out
#SBATCH --error=/data/users2/yxiao11/model/ssl/%x_%j.err

cd /data/users2/yxiao11/
source .bashrc
source activate p37

cd /data/users2/yxiao11/model/ssl

# # Make sure the GPUs are visible
# export CUDA_VISIBLE_DEVICES=0,1,2,3
NGPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}"
echo "Slurm allocated GPUs on node: $NGPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
  --data_dir ../datas/imagenet100/ --out_dir ./runs/imagenet100/ --batch_size 1024 --img_size 224 \
   --amp --num_workers 8 --projector 1024-1024-1024 --epochs 500 --method jdrx --num_subsets 16 \
   --sync_jdrx_stats --image_net

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar100/ --out_dir ./runs/cifar100/ --batch_size 1024 --img_size 32 \
#   --amp --num_workers 4 --projector 2048-2048-256 --epochs 400 --method byol --byol_pred_hidden_dim 1024 \

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/imagenet100/ --out_dir ./runs/imagenet100/ --batch_size 1024 --img_size 224 \
#   --amp --num_workers 8 --projector 1024-1024-1024 --epochs 500 --method vicreg \
#   --image_net

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/imagenet100/ --out_dir ./runs/imagenet100/ --batch_size 1024 --img_size 224 \
#   --amp --num_workers 8 --projector 2048-2048-256 --epochs 500 \
#   --method byol --byol_pred_hidden_dim 2048 --image_net

# # #-------simclr
# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/imagenet100/ --out_dir ./runs/imagenet100/ --batch_size 2048 --img_size 224 \
#   --amp --num_workers 8 --projector 1024-1024-1024 --epochs 500 --method simclr \
#   --temperature 0.2 --image_net


# python main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 1024 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-128 --epochs 400 --method simclr 

# torchrun --nproc_per_node="$NGPUS" linear_eval_ddp.py --data_dir ../datas/imagenet100/ --ckpt ./runs/imagenet100/simclr_ddp8_bs1024_dim1024_ep500/ckpt_latest.pt \
#   --img_size 224 --epochs 100 --batch_size 128 --lr 0.1 --num_workers 8 --amp --out_dir ./runs/imagenet100/simclr_ddp8_bs1024_dim1024_ep500/linear_eval \
#    --method simclr --image_net

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py\
#    --data_dir ../datas/imagenet100/ --out_dir ./runs/imagenet100/ --batch_size 2048 --img_size 224 \
#    --amp --num_workers 8 --projector 1024-1024-1024 --epochs 500 --method jdrx \
#    --num_subsets 16 --sync_jdrx_stats --image_net

# # #-------simclr
# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/imagenet100/ --out_dir ./runs/imagenet100/ --batch_size 2048 --img_size 224 \
#   --amp --num_workers 8 --projector 1024-1024-1024 --epochs 500 --method simclr \
#   --temperature 0.2 --image_net

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/imagenet100/ --out_dir ./runs/imagenet100/ --batch_size 2048 --img_size 224 \
#   --amp --num_workers 8 --projector 1024-1024-1024 --epochs 500 --method barlow \
#   --image_net


# python main_jdrx_ddp.py \
#   --data_dir ../datas/stl10/ --out_dir ./runs/stl10/ --batch_size 2048 --img_size 96 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method jdrx \
#   --num_subsets 256 --stl10

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/stl10/ --out_dir ./runs/stl10/ --batch_size 512 --img_size 96 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method jdrx \
#   --num_subsets 8 --sync_jdrx_stats --stl10

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/stl10/ --out_dir ./runs/stl10/ --batch_size 1024 --img_size 96 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method jdrx \
#   --num_subsets 16 --sync_jdrx_stats --stl10


# python main_jdrx_ddp.py \
#   --data_dir ../datas/stl10/ --out_dir ./runs/stl10/ --batch_size 512 --img_size 96 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method barlow \
#   --stl10

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/stl10/ --out_dir ./runs/stl10/ --batch_size 2048 --img_size 96 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method barlow \
#   --stl10








# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 512 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method simclr \
#   --temperature 0.2 
#-------------
# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar100/ --out_dir ./runs/cifar100/ --batch_size 512 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method jdrx \
#   --num_subsets 8 --sync_jdrx_stats

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar100/ --out_dir ./runs/cifar100/ --batch_size 512 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method barlow \


# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/stl10/ --out_dir ./runs/stl10/ --batch_size 2048 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method jdrx \
#   --num_subsets 32 --sync_jdrx_stats --stl10

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/stl10/ --out_dir ./runs/stl10/ --batch_size 2048 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method barlow --stl10 \


# #-------for batchsize 2048
# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 2048 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-128 --epochs 400 --method jdrx \
#   --num_subsets 32 --sync_jdrx_stats

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 2048 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method barlow \

# #-----jdrx

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 1024 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method jdrx \
#   --num_subsets 8 --sync_jdrx_stats

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 1024 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method jdrx \
#   --num_subsets 4 --sync_jdrx_stats

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 1024 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method jdrx \
#   --num_subsets 2 --sync_jdrx_stats
#------------------------




#-----below barlow
# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 128 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method barlow \


# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 256 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method barlow \


# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar10/ --out_dir ./runs/cifar10/ --batch_size 512 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method barlow \




# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/imagenet100/ --out_dir ./runs/imagenet100/ --batch_size 1024 --img_size 224 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 500 --method barlow --image_net

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/imagenet100/ --out_dir ./runs/imagenet100/ --batch_size 1024 --img_size 224 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 500 --method jdrx \
#   --num_subsets 8 --sync_jdrx_stats --image_net

# # Run the training script
# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar100/ --out_dir ./runs/cifar100/ --batch_size 1024 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method jdrx \
#   --num_subsets 8 --sync_jdrx_stats

# torchrun --standalone --nproc_per_node="$NGPUS" main_jdrx_ddp.py \
#   --data_dir ../datas/cifar100/ --out_dir ./runs/cifar100/ --batch_size 1024 --img_size 32 \
#   --amp --num_workers 4 --projector 1024-1024-1024 --epochs 400 --method barlow \

