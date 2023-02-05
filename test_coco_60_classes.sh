#!/bin/sh
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH -A nuig02
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom020c/Sai/RQ3/logs/test_coco_60_2.txt
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=BEGIN,END
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4 
cd /ichec/work/nuig02/saiabinesh/EdgeNets
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1 python test_detection.py --model espnetv2 --s 2.0 --dataset coco --data-path ./vision_datasets/coco --im-size 300 --weights-test ./results_detection/model_espnetv2_coco/s_2.0_sch_hybrid_im_300/20220103-181009/espnetv2_2.0_checkpoint.pth.tar
