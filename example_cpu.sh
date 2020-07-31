export CUDA_VISIBLE_DEVICES=""

bazel-bin/lingvo/trainer --logdir=librispeech_logs4 \
                         --mode=sync --run_locally=cpu \
                         --model=asr.librispeech_ctc.Librispeech960Base \
                         --logtostderr
