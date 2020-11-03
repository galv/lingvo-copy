input_tarball=$1 ## /home/anjali/data/peoples/tmp/dev_1.tar.gz
output_base="$HOME/data/peoples/dataset"
output_dir="${output_base}/${2}"
num_shards=${3:-1}

bazel run //lingvo/tools:create_peoples_speech_asr_features -- --logtostderr \
    --generate_tfrecords \
    --input_tarball=${input_tarball} \
    --input_text=${input_tarball/tar.gz/csv} \
    --num_shards 1 \
    --shard_id 0 \
    --output_range_begin 0 \
    --output_range_end 1 \
    --num_output_shards ${num_shards} \
    --transcripts_filepath "${output_dir}.txt" \
    --output_template "${output_dir}.tfrecords-%5.5d-of-%5.5d" 2>&1 | tee featurize.log
