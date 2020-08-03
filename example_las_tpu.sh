export CUDA_VISIBLE_DEVICES=""

# ctpu up -tpu-only -tpu-size v3-8 --tf-version 2.2
bazel-bin/lingvo/trainer --logdir=logs/galvez/tpu_las_1a \
       --mode=sync \
       --run_locally=tpu \
       --model=asr.librispeech.Librispeech960Base \
       --logtostderr \
       --tpu=grpc://10.240.1.2:8470
