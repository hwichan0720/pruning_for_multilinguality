ROOT=../..
SRC=$ROOT/src
GPU=$1
MODEL_NAME_OR_PATH=$2
TGT_LANG=$3

TASK=marc2
TEMP_LANG=en
BATCH_SIZE=8
MODE=debug
DATA=$ROOT/data/MARC-2/$TGT_LANG

mkdir -p $MODEL_NAME_OR_PATH/$TASK

CUDA_VISIBLE_DEVICES=$GPU python $SRC/evaluate.py \
    --model_name $MODEL_NAME_OR_PATH \
    --data_path $DATA \
    --labels negative positive \
    --target_lang $TGT_LANG \
    --task $TASK \
    --split test \
    --target_template_lang $TEMP_LANG \
    --batch_size $BATCH_SIZE \
    --seed 42 \
    --mode $MODE \
    --save_path $MODEL_NAME_OR_PATH/$TASK/$TGT_LANG.$MODE