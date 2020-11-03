input_tarball=$1
output_base="$HOME/data/peoples/dataset"
output_dir="${output_base}/${2}"
num_shards=${3:-1}
last_shard=$((num_shards - 1))

# for shard_id in $(seq $last_shard -1 0); do
for shard_id in $(seq 2 $last_shard); do

    bazel run //lingvo/tools:create_asr_features -- --logtostderr \
        --generate_tfrecords \
        --input_tarball=${input_tarball} \
        --input_text=${input_tarball/tar.gz/csv} \
        --shard_id ${shard_id} \
        --output_range_begin ${shard_id} \
        --output_range_end $((shard_id + 1)) \
        --num_output_shards ${num_shards} \
        --transcripts_filepath "${output_dir}.txt" \
        --output_template "${output_dir}.tfrecords-%5.5d-of-%5.5d" 2>&1 | tee featurize.log.train.${shard_id}

    fname="${output_dir}.tfrecords-0000${shard_id}-of-00500"
    gsutil -m cp $fname gs://the-peoples-speech-west-europe/PeoplesSpeech/v0.5/train/
    rm $fname

done
