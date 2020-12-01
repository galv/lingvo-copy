#!/bin/bash

python galvasr2/align/align.py \
       --stt-model-dir third_party/DSAlign/models/en \
       --force \
       --audio sample_alignment_data/Highway_and_Hedges_Outreach_Ministries_Show_-_Show_49.mp3 \
       --script sample_alignment_data/Highway_and_Hedges_Outreach_Ministries_Show_-_Show_49.txt \
       --aligned sample_alignment_data/aligned.json \
       --tlog sample_alignment_data/transcript.log \

       
       --per-document-lm

# --align-shrink-fraction 0 --align-similarity-algo "levenshtein" --output-wng --output-jaro_winkler --output-editex --output-levenshtein --output-mra --output-hamming --output-cer --output-wer --output-sws --output-mlen --output-tlen --output-pretty \
