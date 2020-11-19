# Run as `./featurize.sh /mnt/disks/datasets/raw v0.5.3 16`
# Run as `./featurize.sh gs://the-peoples-speech-west-europe/peoples-speech-v0.5 v0.5.2 16`
# input tarball, output location, num_processing_shards (must divide 512)

# output_base="$HOME/data/peoples/dataset"
# output_base="gs://the-peoples-speech-west-europe/PeoplesSpeech"
# output_base="/mnt/disks/dataset/feats"
output_base="/mnt/disks/data/peoples-speech-v0.7/feats"

function launch_feat() {
    shard_id=$1
    range_step=$((num_output_shards / num_proc_shards))
    bazel run //lingvo/tools:create_peoples_speech_asr_features -- --logtostderr \
        --generate_tfrecords \
        --input_tarball=${input_tarball} \
        --input_text=${input_tarball/tar.gz/csv} \
        --num_shards ${num_proc_shards} \
        --shard_id ${shard_id} \
        --output_range_begin $((range_step * shard_id)) \
        --output_range_end $((range_step * shard_id + range_step)) \
        --num_output_shards ${num_output_shards} \
        --transcripts_filepath "${output_dir}.txt" \
        --output_template "${output_dir}.tfrecords-%5.5d-of-%5.5d" 2>&1 | tee feat.${shard_id}.log
}
export -f launch_feat

export input_tarball="${1}/development.tar.gz"
export output_dir="${output_base}/devtest/dev"
export num_proc_shards=1
export num_output_shards=1
launch_feat 0

export input_tarball="${1}/test.tar.gz"
export output_dir="${output_base}/devtest/test"
export num_proc_shards=1
export num_output_shards=1
launch_feat 0

export input_tarball="${1}/train.tar.gz"
export output_dir="${output_base}/train/train"
export num_proc_shards=${2:-16}
last_shard=$((num_proc_shards - 1))
export num_output_shards=512
parallel -j $num_proc_shards launch_feat ::: $(seq 0 $last_shard)

#./featurize.sh /mnt/disks/dataset/raw v0.5.3
#gsutil -m rsync /mnt/disks/dataset/feats/v0.5.3/train gs://the-peoples-speech-west-europe/PeoplesSpeech/v0.5.2/train