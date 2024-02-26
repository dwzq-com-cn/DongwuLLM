from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import argparse
import time

def generate_and_save(prefix_texts, llm, sampling_params, args):
    outputs = llm.generate(prefix_texts, sampling_params)
    generated_texts = [output.outputs[0].text for output in outputs]
    with open(args.save_prefix + '.jsonl', 'a+') as f:
        for prefix, text in zip(prefix_texts, generated_texts):
            f.write(json.dumps({'prefix':prefix, 'text':prefix + text}, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argparse For LLM Inference")
    # args related to data
    parser.add_argument('--data-path', type=str, help='path of jsonl data which provies prefix text')
    parser.add_argument('--text-len', type=int, default=4096, help='len of full text')
    parser.add_argument('--prefix-len', type=int, default=100, help='len of prefix')
    parser.add_argument('--save-prefix', type=str, help='prefix path to save the generated data')
    parser.add_argument('--block-size', type=int, default=100, help='the block to save the generated data')
    parser.add_argument('--start-samples', type=int, default=100000, help='lines to start generate')
    parser.add_argument('--end-samples', type=int, default=200000, help='lines to stop generate')

    # args related to model
    parser.add_argument('--model-path', type=str, help='path to load the model')
    parser.add_argument('--temp', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--rep-penalty', type=float, default=2)
    parser.add_argument('--freq-penalty', type=float, default=0.5)
    # parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--tp-size', type=int, default=4)

    args = parser.parse_args()

    # print(f"This script will generate {args.gen_samples * args.text_len / 1e9} B tokens, \
    # and {args.gen_samples * (args.text_len - args.prefix_len) / 1e9} B tokens are generate by LLM")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(model=args.model_path, trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=args.tp_size)
    sampling_params = SamplingParams(temperature=args.temp, 
                                     top_p=args.top_p, 
                                     max_tokens=args.text_len, 
                                    #  ignore_eos=True,
                                     frequency_penalty=args.freq_penalty,
                                     repetition_penalty=args.rep_penalty,
                                     stop=['<|endoftext|>'],
                                     include_stop_str_in_output=True)

    with open(args.data_path, 'r') as f:
        prefix_texts = []
        for i in tqdm(range(0, args.end_samples)):
            text = json.loads(f.readline())['text']
            if args.start_samples > i: continue
            prefix_text = tokenizer.decode(tokenizer(text)['input_ids'][:args.prefix_len], skip_special_tokens=True)
            if prefix_text: prefix_texts.append(prefix_text)
            if i % args.block_size == args.block_size - 1:
                begin_time = time.time()
                generate_and_save(prefix_texts, llm, sampling_params, args)
                end_time = time.time()
                token_speed = args.block_size * args.text_len / (end_time - begin_time) * 3600 * 24 / 1e9
                print(f"Speed: {token_speed} B / day")
                prefix_texts = []
                