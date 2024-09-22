ROOT=../..
SRC=$ROOT/src
GPU=$1
MODEL_NAME=$2
TGT_LANG=$3

TASK=xnli
SAVE_DIR=$ROOT/outputs
N_SAMPLES=100

CUDA_VISIBLE_DEVICES=$GPU python $SRC/lrp2_search.py \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \
    --num_examples $N_SAMPLES \
    --task $TASK \
    --split validation \
    --target_lang $TGT_LANG \
    --source_lang en \
    --seed 0 \
    --batch_size 16 \
    --mode debug \
