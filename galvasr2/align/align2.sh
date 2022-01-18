#!/bin/bash

debug_loglevel=10

for aggressiveness in $(seq 0 3); do
    out_dir=sample_alignment_data2/output-generic-lm/${aggressiveness}
    mkdir -p ${out_dir}
    python galvasr2/align/align.py \
       --stt-model-dir third_party/DSAlign/models/en \
       --force \
       --audio sample_alignment_data2/021A-C0897X0104XX-AAZZP0.wav \
       --script sample_alignment_data2/021A-C0897X0104XX-AAZZP0.txt \
       --aligned ${out_dir}/aligned.json \
       --tlog ${out_dir}/transcript.log \
       --audio-vad-aggressiveness $aggressiveness \
       --loglevel ${debug_loglevel} \
       --output-wng --output-jaro_winkler --output-editex --output-levenshtein --output-mra --output-hamming --output-cer --output-wer --output-sws --output-mlen --output-tlen
done
