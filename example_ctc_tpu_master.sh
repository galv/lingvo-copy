export OPENBLAS_NUM_THREADS="1"
export MKL_NUM_THREADS="1"

#$1 = tpu/gpu (def tpu)
#$2 = train + eval_dev or train only (def train)
#$3 = class_name (def Librispeech960Base)

HOME_BASE="/home/anjali/data/mlcommons/librispeech/models/wer"
GS_BASE="gs://the-peoples-speech-west-europe/ag/ctc_librispeech/training_logs"

DATE=$(date '+log_%Y_%m_%d_%H')
FLDRDATE=$(date '+%m%d/%H%M')
CLS=$3

# if tpu- in $1
if [[ "$1" == *"tpu-"* ]]; then
    LOGDIR="${GS_BASE}/${FLDRDATE}/${CLS}"

    name=$1
    ip_addr=$(../ctpu st -name ${name} --details | grep "TPU IP" | grep -oP "10.*")

    if [ -z "$ip_addr" ]; then
        echo "Couldnt find TPU, creating a new one"
        ../ctpu up -name ${name} -tpu-only -tpu-size v3-8 -tf-version 2.2 -preemptible
        ../ctpu st -name ${name} --details | grep "TPU IP" | grep -oP "10.*"
        ip_addr=$(../ctpu st -name ${name} --details | grep "TPU IP" | grep -oP "10.*")
    fi

    TPUIP=$ip_addr

elif [ $1 == "gpu" ]; then
    export CUDA_VISIBLE_DEVICES="0"
    # AG TODO: add min and hr to the folder name
    # LOGDIR="/home/anjali/data/librispeech_models/wer/${DATE}"
    LOGDIR="${HOME_BASE}/${DATE}"
else
    export CUDA_VISIBLE_DEVICES=""
    LOGDIR="${GS_BASE}/${FLDRDATE}/${CLS}"
    TPUIP=$5
fi

if [ $2 == "decode" ]; then
    OPERATION="decoder_dev"
elif [ $2 == "trainer" ]; then
    OPERATION="trainer_client"
else
    OPERATION="executor_tpu"
fi

bazel run //lingvo:trainer -- --logdir=${LOGDIR} \
    --mode=sync \
    --model=asr.librispeech_ctc.${CLS} \
    --logtostderr \
    --tpu=grpc://${TPUIP}:8470 \
    --job=$OPERATION 2>&1 | tee logs/${CLS}_${DATE}.log
