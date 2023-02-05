#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH -A ngcom027c
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom027c/logs/test_imagenet_classification_copied_top_30.sh
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=ALL
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4
cd /ichec/work/nuig02/saiabinesh/EdgeNets
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1 python -u test_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets_size_corrected/imgnet_top_30 --weights results_classification_main/model_espnetv2_imagenet/aug_0.2_1.0/s_2.0_inp_224_sch_hybrid/20220907-083926//espnetv2_2.0_checkpoint.pth.tar --num-classes 30
