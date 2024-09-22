ROOT=../..
SRC=$ROOT/src
GPU=$1
MODEL_NAME=$2
LANG=$3

SAVE_DIR=$ROOT/outputs
N_SAMPLES=100

CUDA_VISIBLE_DEVICES=$GPU python $SRC/get_lang_vector.py \
    --model_name $MODEL_NAME \
    --save_dir $SAVE_DIR \
    --num_examples $N_SAMPLES \
    --lang $LANG