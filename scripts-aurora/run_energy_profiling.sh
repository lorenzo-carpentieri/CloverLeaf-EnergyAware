# num of runs
NUM_RUNS=$1

# Values fixed for Intel Max 1550
MIN_FREQ_MHZ=$2
MAX_FREQ_MHZ=$3
STEP=$4 # Use multiple of 50 since this is the step for Intel Max 1550. 
NUM_RANKS=$5
NUM_PROCESS=$6
EXE=$7

INPUT_PATH=/home/lcarpent/energy-workspace/multinode-apps/cloverleaf_sycl/input_file/clover_bm8_short.in

SCRIPT_DIR=$(dirname "$(readlink -f "$0")") 
BUILD_DIR=$SCRIPT_DIR/../build 
LOG_DIR="$SCRIPT_DIR/../logs/"


echo $LOG_DIR
echo $BUILD_DIR
echo $SCRIPT_DIR
if [ ! -d "${LOG_DIR}" ]; then
    # Create the directory
    mkdir -p "${LOG_DIR}"
    echo "Directory created: ${LOG_DIR}"
else
    echo "Directory already exists: ${LOG_DIR}"
    rm -rf ${LOG_DIR}/*
fi


for ((freq=$MIN_FREQ_MHZ; freq<=$MAX_FREQ_MHZ; freq+=$STEP)); do
    echo "----------------------------------------"
    echo "Running frequency: ${freq} MHz"
    echo "----------------------------------------"

    for ((i=0; i<$NUM_RUNS; i++)); do
        # Create difrecotory for log files
        if [ ! -d "${LOG_DIR}/freq-${freq}/run-${i}/" ]; then
            # Create the directory
            mkdir -p "${LOG_DIR}/freq-${freq}/run-${i}/"
            echo "Directory created: ${LOG_DIR}/freq-${freq}/run-${i}/"
        else
            echo "Directory already exists: ${LOG_DIR}/freq-${freq}/run-${i}/"
            rm -rf ${LOG_DIR}/*
        fi

        echo "Run $i at ${freq} MHz"
        # Change freq with geopmwrite


        for ((j=0; j<$NUM_PROCESS; j++)); do
            geopmwrite GPU_CORE_FREQUENCY_MAX_CONTROL gpu_chip ${j} 1600e6
            geopmwrite GPU_CORE_FREQUENCY_MIN_CONTROL gpu_chip ${j} 1600e6 # Set the freq to max for min freq and max freq so that the next command are always fine
            geopmwrite GPU_CORE_FREQUENCY_MIN_CONTROL gpu_chip ${j} ${freq}e6
            geopmwrite GPU_CORE_FREQUENCY_MAX_CONTROL gpu_chip ${j} ${freq}e6
            echo "GPU ${j} Energy: "
            geopmread  GPU_CORE_ENERGY gpu_chip ${j} 
        done

        mpiexec -n ${NUM_RANKS} -ppn ${NUM_PROCESS} "${BUILD_DIR}/${EXE}" --file ${INPUT_PATH} --gpu-bind > "${LOG_DIR}/freq-${freq}/run-${i}/output.log"
        for ((j=0; j<$NUM_PROCESS; j++)); do
            echo "GPU ${j} Energy: "
            geopmread  GPU_CORE_ENERGY gpu_chip ${j} 
        done
    done
done


