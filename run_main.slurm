#!/bin/bash
#SBATCH --job-name=in5550_eval_models
#SBATCH --account=ec30
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal

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

cd /fp/homes01/u01/ec-liseche/exam_IN5550

python3 eval.py --method xlmr --dataset /fp/homes01/u01/ec-liseche/exam_IN5550/data/test.jsonl --path /cluster/work/projects/ec30/liseche/xlm-roberta-base_gold-train_gold_dev
