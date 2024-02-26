declare -A model_dict
model_dict["deepseek"]="/PATH/TO/deepseek-llm-67b-base"
model_dict["qwen"]="/PATH/TO/Qwen-72B"

DATA_PATH=<Path_TO_YOUR_DATA>
START=0
END=1800000
MODEL_NAME=deepseek

MODEL_PATH=${model_dict[${MODEL_NAME}]}
DATA_NAME=$(basename $(dirname $DATA_PATH))
DATA_ID=$([[ $DATA_PATH =~ ([0-9]+)\.jsonl$ ]] && echo ${BASH_REMATCH[1]})
STARTW=$(( START / 10000 ))
ENDW=$(( END / 10000 ))


python infer.py \
    --data-path ${DATA_PATH}  \
    --save-prefix /PATH/TO/GENERATED/DATA/${MODEL_NAME}/${DATA_NAME}_${DATA_ID}_${STARTW}-${ENDW}w.jsonl \
    --block-size 2048 \
    --tp-size 8 \
    --model-path ${MODEL_PATH} \
    --start-samples ${START} \
    --end-samples ${END} \
    --rep-penalty 2 \
    --freq-penalty 0.5 \
    --prefix-len 128