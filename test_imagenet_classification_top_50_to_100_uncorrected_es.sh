#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH -A ngcom027c
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom027c/logs/test_imagenet_classification_top_50_to_100_uncorrected_es.sh
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=ALL
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4
export OMP_NUM_THREADS=1
cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_20 --weights results_classification_main/model_espnetv2_imagenet/aug_0.2_1.0/s_2.0_inp_224_sch_hybrid/20220801-232058//espnetv2_2.0_checkpoint.pth.tar --num-classes 20 > test_top_20_uncorrected.txt

cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_50 --weights results_classification_main/model_espnetv2_imagenet/aug_0.2_1.0/s_2.0_inp_224_sch_hybrid/20220803-111751//espnetv2_2.0_checkpoint.pth.tar
 --num-classes 50 > test_top_50_uncorrected.txt

cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_60 --weights results_classification_main/model_espnetv2_imagenet/aug_0.2_1.0/s_2.0_inp_224_sch_hybrid/20220803-140621//espnetv2_2.0_checkpoint.pth.tar
 --num-classes 60 > test_top_60_uncorrected.txt

cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_70 --weights results_classification_main/model_espnetv2_imagenet/aug_0.2_1.0/s_2.0_inp_224_sch_hybrid/20220803-173432//espnetv2_2.0_checkpoint.pth.tar
 --num-classes 70 > test_top_70_uncorrected.txt


cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification_args.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_80 --weights results_classification_main/model_espnetv2_imagenet/aug_0.2_1.0/s_2.0_inp_224_sch_hybrid/20220803-212642//espnetv2_2.0_checkpoint.pth.tar
 --num-classes 80 >test_top_80_uncorrected.txt