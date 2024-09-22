ROOT=../..
SRC=$ROOT/src
GPU=$1
MODEL_NAME=$2
S_RATIO=$3
DATA_TYPE=$4
SRC_LANG=$5
TGT_LANG=$6

S_TYPE=unstructured
SAVE_MODEL=$ROOT/outputs
N_SAMPLES=100
SHOTS=4

CUDA_VISIBLE_DEVICES=$GPU python $SRC/prune_mt.py \
    --model_name $MODEL_NAME \
    --sparsity_ratio $S_RATIO \
    --sparsity_type $S_TYPE \
    --source_lang $SRC_LANG \
    --target_lang $TGT_LANG \
    --save_model $SAVE_MODEL \
    --nsamples $N_SAMPLES \
    --shots $SHOTS \
    --data_type $DATA_TYPE