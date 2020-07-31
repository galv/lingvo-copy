# python -m lingvo.train --ctc_model=PLPCoefficients \
#        --abc 1 \
#        --xyz 2

# bazel run //lingvo/train --model

export CUDA_VISIBLE_DEVICES=""

# ctpu up -tpu-only -tpu-size v3-8 --tf-version 2.2

# gs://the-peoples-speech-west-europe/Librispeech/logs/galvez/tpu_1a \
bazel-bin/lingvo/trainer --logdir=logs/galvez/tpu_1d \
                         --mode=sync \
                         --model=asr.librispeech_ctc.Librispeech960Base \
                         --logtostderr \
                         --tpu=grpc://10.240.1.2:8470
