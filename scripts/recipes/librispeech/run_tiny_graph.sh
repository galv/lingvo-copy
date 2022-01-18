#!/bin/bash

set -euo pipefail

stage=0

work_dir=tiny_graph_work_2a
langdir=$work_dir/data/lang_char_tgsmall

mkdir -p $work_dir/data/local/dict_char

if [ $stage -le 0 ]; then
    galvasr_tokenize_words --in_words_txt local/tiny_graph/words.txt \
                           --in_units_txt local/tokens_cat.txt \
                           --out_spelling_txt $work_dir/data/local/dict_char/lexicon.txt \
                           --out_spelling_numbers_txt $work_dir/data/local/dict_char/lexicon_numbers.txt \
                           --out_units_txt $work_dir/data/local/dict_char/units.txt \
                           --space_char "<space>"
    steps/ctc_compile_dict_token.sh --dict-type char \
                                    --space-char "<space>" \
                                    $work_dir/data/local/dict_char \
                                    $work_dir/data/local/lang_char_tmp $work_dir/data/lang_char
    local/format_lms.sh --src-dir $work_dir/data/lang_char $work_dir/data/local/lm

    fsttablecompose ${langdir}/L.fst $langdir/G.fst | fstdeterminizestar --use-log=true | \
        fstminimizeencoded | fstarcsort --sort_type=ilabel > $langdir/LG.fst || exit 1;
    # Why isn't this determinized and then minimized? Unclear.
    fsttablecompose ${langdir}/T.fst $langdir/LG.fst > $langdir/TLG.fst || exit 1;
    fstconvert --fst_type=const $langdir/TLG.fst $langdir/TLG_const.fst

    # fstdraw --isymbols=$work_dir/data/lang_char/tokens.txt --osymbols=$work_dir/data/lang_char/tokens.txt $work_dir/data/lang_char/T.fst | dot -Tpdf > T.pdf
    # fstdraw --isymbols=$work_dir/data/lang_char/tokens.txt --osymbols=$work_dir/data/lang_char/words.txt $work_dir/data/lang_char/L.fst | dot -Tpdf > L.pdf
    # fstdraw --isymbols=$work_dir/data/lang_char/tokens.txt --osymbols=$work_dir/data/lang_char/words.txt $langdir/TLG.fst | dot -Tpdf > TLG.pdf


    # fsttablecompose ${langdir}/T.fst $langdir/L.fst | fstdeterminizestar --use-log=true | \
    #     fstminimizeencoded | fstarcsort --sort_type=ilabel > $langdir/TL.fst || exit 1;
    # fstconvert --fst_type=const $langdir/TL.fst $langdir/TL_const.fst
fi
