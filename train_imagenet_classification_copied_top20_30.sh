#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A nuig02
#SBATCH -p GpuQ
#SBATCH -o /ichec/work/ngcom027c/logs/train_imagenet_classification_copied_top20_30.sh
#SBATCH --mail-user=s.natarajan3@nuigalway.ie
#SBATCH --mail-type=ALL
module load cuda/11.3 gcc conda/2
source activate /ichec/work/nuig02/saiabinesh/pytorch4
cd /ichec/work/ngcom027c/Sai/general_scripts
python create_temp_dataset_by_copying_size_n_imgnt_new.py /ichec/work/ngcom027c/Datasets/ILSVRC/Data/CLS-LOC 10
python copy_training_files_n_copies.py /ichec/work/ngcom027c/Datasets/imgnet_subsets_size_corrected/imgnet_top_20/train 50
cd /ichec/work/nuig02/saiabinesh/EdgeNets
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets_size_corrected/imgnet_top_20 --scale 0.2 1.0 --scheduler hybrid --lr 0.01 --clr-max 20 --batch-size 144 --epochs 100
cd /ichec/work/ngcom027c/Sai/general_scripts
python create_temp_dataset_by_copying_size_n_imgnt_new.py /ichec/work/ngcom027c/Datasets/ILSVRC/Data/CLS-LOC 30
python copy_training_files_n_copies.py /ichec/work/ngcom027c/Datasets/imgnet_subsets_size_corrected/imgnet_top_30/train 33
cd /ichec/work/nuig02/saiabinesh/EdgeNets
export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1 python -u train_classification.py --model espnetv2 --s 2.0 --dataset imagenet --data /ichec/work/ngcom027c/Datasets/imgnet_subsets_size_corrected/imgnet_top_30 --scale 0.2 1.0 --scheduler hybrid --lr 0.01 --clr-max 20 --batch-size 144 --epochs 100
