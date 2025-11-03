#!/bin/bash
#PBS -N cloverleaf_freq_scaling
#PBS -A  EnergyOpt_PhaseFreq
#PBS -l select=1
#PBS -l walltime=00:10:00
#PBS -l filesystems=home
#PBS -o /home/lcarpent/energy-workspace/multinode-apps/CloverLeaf/pbs-out/output.txt
#PBS -e /home/lcarpent/energy-workspace/multinode-apps/CloverLeaf/pbs-out/error.txt
#PBS -q debug


# Load required modules or source oneAPI environment
source /home/lcarpent/energy-workspace/multinode-apps/cloverleaf_sycl/scripts-aurora/env/set_env.sh
export  ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# Set parameters for the energy profiling script
# NUM_RUNS=3
# STEP=200
# MIN_FREQ=200
# MAX_FREQ=1600
# NUM_RANKS=6
# NUM_PROCESS=6 

NUM_RUNS=1
STEP=1400
MIN_FREQ=200
MAX_FREQ=1600
NUM_RANKS=3
NUM_PROCESS=3
EXE="sycl-usm-cloverleaf"

RUN_SCRIPT_DIR="/home/lcarpent/energy-workspace/multinode-apps/CloverLeaf-EnergyAware/scripts-aurora/"

${RUN_SCRIPT_DIR}/run_energy_profiling.sh $NUM_RUNS $MIN_FREQ $MAX_FREQ $STEP $NUM_RANKS $NUM_PROCESS $EXE