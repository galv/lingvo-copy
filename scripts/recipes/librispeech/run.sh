#!/bin/bash

set -euo pipefail

source ./path.sh

lm_url=www.openslr.org/resources/11
work_dir=work_1a

stage=2

mkdir -p $work_dir

# bazel build @openfst//...
# bazel build third_party/kaldi:arpa2fst

if [ $stage -le 1 ]; then
   local/download_lm.sh $lm_url $work_dir/data/local/lm
fi

langdir=$work_dir/data/lang_char_tgsmall
if [ $stage -le 2 ]; then
    # local/prepare_dict_ctc.sh $work_dir/data/local/lm $work_dir/data/local/dict_phn
    # steps/ctc_compile_dict_token.sh $work_dir/data/local/dict_phn $work_dir/data/local/lang_phn_tmp $work_dir/data/lang_phn
    mkdir -p $work_dir/data/local/dict_char
    galvasr_tokenize_words --words_txt $work_dir/data/local/lm/librispeech-vocab.txt \
                           --spelling_txt $work_dir/data/local/dict_char/lexicon.txt \
                           --spelling_numbers_txt $work_dir/data/local/dict_char/lexicon_numbers.txt \
                           --units_txt $work_dir/data/local/dict_char/units.txt
    steps/ctc_compile_dict_token.sh --dict-type nchar \
                                    --space-char OPTIONAL_SILENCE_AFTER_WORD_SPACE_IS_UNUSED \
                                    $work_dir/data/local/dict_char \
                                    $work_dir/data/local/lang_char_tmp $work_dir/data/lang_char
    # fstdraw --isymbols=work_1a/data/lang_char/tokens.txt --osymbols=work_1a/data/lang_char/words.txt work_1a/data/lang_char/L.fst | dot -Tpdf > L.pdf
    local/format_lms.sh --src-dir $work_dir/data/lang_char $work_dir/data/local/lm

    fsttablecompose ${langdir}/L.fst $langdir/G.fst | fstdeterminizestar --use-log=true | \
        fstminimizeencoded | fstarcsort --sort_type=ilabel > $langdir/LG.fst || exit 1;
    # Why isn't this determinized and then minimized? Unclear.
    fsttablecompose ${langdir}/T.fst $langdir/LG.fst > $langdir/TLG.fst || exit 1;

    fstconvert --fst_type=const $langdir/TLG.fst $langdir/TLG_const.fst
fi

exit 0

if [ $stage -le 3 ]; then
    galvasr_latgen_faster $langdir/TLG_const.fst blah.tfrecord ark,t:
fi
