#!/bin/bash
#SBATCH --job-name=nqx-kernel-bench
#SBATCH --partition=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=lp82
#SBATCH --output=bench_kernel_%j.out

set -eu
cd /capstor/scratch/cscs/cklin/programme/dist_nequix/nequix
source ../.venv/bin/activate

# openequivariance_extjax dynamically links libcudart/libnvrtc from the pip
# nvidia wheels; put them on the loader path
export LD_LIBRARY_PATH="$(ls -d ../.venv/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"

# correctness: pytest kernel suite, then full-model check on 4 GPUs
python -m pytest tests/test_distributed_kernel.py -q
python scripts/bench_distributed.py --n-devices 4 --size 4 --steps 5 --check --kernel

# kernel scaling sweep (skin=0.3 default); e3nn numbers for these configs are
# in bench_distributed_3641837.out
for size in 10 14; do
    for nd in 1 4; do
        python scripts/bench_distributed.py \
            --n-devices "$nd" --size "$size" --steps 30 --kernel
    done
done
