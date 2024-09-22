ROOT=../..
SRC=$ROOT/src
GPU=$1
MODEL_NAME=$2
SRC_LANG=$3
TGT_LANG=$4

TASK_NAME=mt
SAVE_DIR=$ROOT/outputs
NUM_SHOTS=4
NUM_EXAMPLES=100

CUDA_VISIBLE_DEVICES=$GPU python $SRC/hidden_states_analysis.py \
    --model_name $MODEL_NAME \
    --task_name $TASK_NAME \
    --save_dir $SAVE_DIR \
    --num_examples $NUM_EXAMPLES \
    --num_shots $NUM_SHOTS \
    --src_lang $SRC_LANG \
    --tgt_lang $TGT_LANG 