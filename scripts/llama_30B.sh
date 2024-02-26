#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=<YOUR_MASTER_ADDR>
MASTER_PORT=6000
NNODES=4
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=<YOUR_CHECKPOINT_PATH>
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
    --pipeline-model-parallel-size 2 \
    --sequence-parallel \
    --num-layers 36 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 12 \
    --global-batch-size 2400 \
    --lr 2.5e-5 \
    --train-samples 4800000 \
    --lr-decay-samples 4672000 \
    --lr-warmup-samples 12800 \
    --lr-decay-style cosine \
    --min-lr 2.5e-5 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --bf16 \
    --log-interval 10 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --no-position-embedding \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --no-masked-softmax-fusion \
    --swiglu \
    --initial-loss-scale 131072 \
    --log-throughput \
    --use-flash-attn \
    --use-distributed-optimizer \
    --attention-softmax-in-fp32 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $VOCAB_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10 \
    --eval-interval 3000 \
    --eval-iters 1
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --tensorboard-dir $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH


