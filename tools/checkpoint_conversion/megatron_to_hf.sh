python tools/checkpoint_conversion/llama_checkpoint_conversion.py \
--convert_checkpoint_from_megatron_to_transformers \
--load_path "/data/Megatron-LM-main/ckpts/showcai-13b-tp1-distill" \
--save_path "/data/hf_models/showcai-13b-hf-distill-6000" \
--target_params_dtype "bf16" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path "/data/Megatron-LM-repo"