#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A ngcom027c
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom027c/logs/train_imagenet_classification_top10_to_40_uncorrected_es.sh
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=ALL
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4
export OMP_NUM_THREADS=1
cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_10 --weights results_classification_main/model_espnetv2_imagenet/aug_0.2_1.0/s_2.0_inp_224_sch_hybrid/20220801-225532//espnetv2_2.0_checkpoint.pth.tar --num-classes 10 > test_top_10_uncorrected.txt

cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_20 --weights results_classification_main/model_espnetv2_imagenet/aug_0.2_1.0/s_2.0_inp_224_sch_hybrid/20220801-232058//espnetv2_2.0_checkpoint.pth.tar --num-classes 20 > test_top_20_uncorrected.txt

cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_30 --weights results_classification_main/model_espnetv2_imagenet/aug_0.2_1.0/s_2.0_inp_224_sch_hybrid/20220802-001208//espnetv2_2.0_checkpoint.pth.tar --num-classes 30 > test_top_30_uncorrected.txt


cd /ichec/work/nuig02/saiabinesh/EdgeNets
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets/imgnet_top_40 --weights results_classification_main/model_espnetv2_imagenet/aug_0.2_1.0/s_2.0_inp_224_sch_hybrid/20220802-015524//espnetv2_2.0_checkpoint.pth.tar --num-classes 40 >test_top_40_uncorrected.txt