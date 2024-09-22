ROOT=../..
SRC=$ROOT/src
HYP_PATH1=$1
HYP_PATH2=$2

OUTPUT_DIR=./
mkdir $OUTPUT_DIR

python $SRC/convert_tsv_to_text.py \
    --path1 $HYP_PATH1 \
    --path2 $HYP_PATH2 \
    --output_dir $OUTPUT_DIR

$SRC/mosesdecoder/scripts/analysis/bootstrap-hypothesis-difference-significance.pl $OUTPUT_DIR/hyp1.txt $OUTPUT_DIR/hyp2.txt $OUTPUT_DIR/ref.txt
