#!/bin/bash
#SBATCH --job-name=in5550_training_gpu_epoch1
#SBATCH --account=ec30
#SBATCH --time=02:30:00           # Adjusted for a longer running time
#SBATCH --partition=accel         # To use the accelerator (GPU) nodes
#SBATCH --gpus=1                  # Request 1 GPU
#SBATCH --cpus-per-task=32        # Number of CPU cores per task
#SBATCH --mem=250G                # Total amount of RAM
#SBATCH --nodes=1

module purge
module use -a /fp/projects01/ec30/software/easybuild/modules/all/
module load nlpl-nlptools/01-foss-2022b-Python-3.10.8
module load nlpl-pytorch/2.1.2-foss-2022b-cuda-12.0.0-Python-3.10.8
module load nlpl-gensim/4.3.2-foss-2022b-Python-3.10.8
module load nlpl-transformers/4.35.2-foss-2022b-Python-3.10.8
module load nlpl-nlptools/01-foss-2022b-Python-3.10.8
module load nlpl-transformers/4.38.2-foss-2022b-Python-3.10.8
module load nlpl-llmtools/03-foss-2022b-Python-3.10.8
module load nlpl-sentencepiece/0.1.99-foss-2022b-Python-3.10.8
module load nlpl-torchmetrics/1.2.1-foss-2022b-Python-3.10.8
cd /fp/homes01/u01/ec-liseche/exam_IN5550/roberta-based

python3 main.py
