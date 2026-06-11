#!/bin/bash
#SBATCH --job-name=nqx-dist-bench
#SBATCH --partition=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=lp82
#SBATCH --output=bench_distributed_%j.out

set -eu
cd /capstor/scratch/cscs/cklin/programme/dist_nequix/nequix
source ../.venv/bin/activate

# correctness check on real GPUs (4-device vs single-device), then scaling sweep
python scripts/bench_distributed.py --n-devices 4 --size 4 --steps 5 --check

for size in 10 14; do
    for nd in 1 4; do
        for skin in 0.0 0.3; do
            python scripts/bench_distributed.py \
                --n-devices "$nd" --size "$size" --steps 30 --skin "$skin"
        done
    done
done
