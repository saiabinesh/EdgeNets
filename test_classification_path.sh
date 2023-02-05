#!/bin/sh
#SBATCH --time=00:09:00
#SBATCH --nodes=1
#SBATCH -A ngcom027c
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/nuig02/saiabinesh/logs/test_classification_path.sh
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=ALL
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4
export OMP_NUM_THREADS=1
cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_50 --logs-file /ichec/work/ngcom027c/logs/train_imagenet_classification_top50_to_100_uncorrected_es.sh --num-classes 50 > test_top_50_uncorrected.txt


