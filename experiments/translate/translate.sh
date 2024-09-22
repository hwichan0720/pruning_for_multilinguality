ROOT=../..
SRC=$ROOT/src
GPU=$1
MODEL_NAME=$2
MODEL_PATH=$3

DATASET=xnli
SAVE_DIR=$ROOT/outputs/$MODEL_PATH/translation/xnli

if [ $MODEL_NAME = $MODEL_PATH ]; then
    SAVE_DIR=$ROOT/outputs/$MODEL_NAME/translation/xnli
else
    SAVE_DIR=$MODEL_PATH/translation/xnli
fi

if [ $MODEL_NAME = "facebook/xglm-2.9B" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python $SRC/translate_dataset_few_shot.py \
        --dataset $DATASET \
        --target_lang "eng_Latn" \
        --starting_batch_size 128 \
        --model_name $MODEL_NAME \
        --loading_model_name $MODEL_PATH \
        --max_length 1024 \
        --max_new_tokens 64 \
        --num_beams 1 \
        --num_return_sequences 1 \
        --precision 32 \
        --eos_token "</s>" \
        --save_dir $SAVE_DIR
elif [ $MODEL_NAME = "ai-forever/mGPT" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python $SRC/translate_dataset_few_shot.py \
        --dataset $DATASET \
        --target_lang "eng_Latn" \
        --starting_batch_size 128 \
        --model_name $MODEL_NAME \
        --loading_model_name $MODEL_PATH \
        --max_length 1024 \
        --max_new_tokens 64 \
        --num_beams 1 \
        --num_return_sequences 1 \
        --precision 32 \
        --eos_token "</s>" \
        --save_dir $SAVE_DIR
elif [ $MODEL_NAME = "bigscience/bloom-3B" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python $SRC/translate_dataset_few_shot.py \
        --dataset $DATASET \
        --target_lang "eng_Latn" \
        --starting_batch_size 128 \
        --model_name $MODEL_NAME \
        --loading_model_name $MODEL_PATH \
        --max_length 1024 \
        --max_new_tokens 64 \
        --num_beams 1 \
        --num_return_sequences 1 \
        --precision 32 \
        --save_dir $SAVE_DIR
elif [ $MODEL_NAME = "meta-llama/Llama-2-7b-hf" ]; then
    CUDA_VISIBLE_DEVICES=$GPU python $SRC/translate_dataset_few_shot.py \
        --dataset $DATASET \
        --target_lang "eng_Latn" \
        --starting_batch_size 128 \
        --model_name $MODEL_NAME \
        --loading_model_name $MODEL_PATH \
        --max_length 1024 \
        --max_new_tokens 64 \
        --num_beams 1 \
        --num_return_sequences 1 \
        --precision 32 \
        --save_dir $SAVE_DIR
fi