#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=<YOUR_MASTER_ADDR>
MASTER_PORT=5284
NNODES=4
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=<PATH_TO_LOAD_CHECKPOINT>
SAVE_PATH=<PATH_TO_SAVE_CHECKPOINT>
VOCAB_FILE=<YOUR_VOCAB_FILE_PATH>
DATA_PATH=<YOUR_DATA_PATH>

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --use-checkpoint-args \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --lr 2e-5 \
    --train-iters 1000 \
    --lr-decay-iters 1000 \
    --lr-warmup-iters 30 \
    --lr-decay-style cosine \
    --min-lr 2e-6 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --bf16 \
    --log-interval 10 \
    --exit-on-missing-checkpoint \
    --no-load-optim \
    --no-load-rng \
    --initial-loss-scale 131072 \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --seq-length 32768 \
    --max-position-embeddings 32768 \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --log-throughput \
    --rotary-seq-len-interpolation-factor 8 \
    --swiglu \
    --recompute-activations \
    --use-distributed-optimizer \
    --use-flash-attn \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $VOCAB_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 250 \
    --eval-interval 10 \
    --eval-iters 1
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $SAVE_PATH \
    --tensorboard-dir $SAVE_PATH \
    --load $CHECKPOINT_PATH


