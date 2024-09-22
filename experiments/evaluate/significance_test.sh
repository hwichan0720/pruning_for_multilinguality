ROOT=../..
SRC=$ROOT/src
PROB_PATH1=$1
PROB_PATH2=$2
TASK=$3
TARGET_LANG=$4
N_LOOPS=1000


if [ $TASK == "marc2" ]; then
  CUDA_VISIBLE_DEVICES=$GPU python $SRC/bootstrap_test.py \
    --prob_path1 $PROB_PATH1 \
    --prob_path2 $PROB_PATH2 \
    --task $TASK \
    --n_loops $N_LOOPS \
    --lang $TARGET_LANG \
    --data_path $ROOT/data/MARC-2/$TARGET_LANG
else
  CUDA_VISIBLE_DEVICES=$GPU python $SRC/bootstrap_test.py \
    --prob_path1 $PROB_PATH1 \
    --prob_path2 $PROB_PATH2 \
    --task $TASK \
    --n_loops $N_LOOPS \
    --lang $TARGET_LANG
fi

