import numpy as np
from megatron.tokenizer.tokenizer import _SentencePieceTokenizer
from megatron.core.datasets import indexed_dataset
import json
# 62189568

# indexmap_filename = "/data/en_zh/all_data_train_indexmap_819200000mns_1234seed.npy"
data_prefix = "/data/v2_data/distill_data/bin_data/v1/lc_text_document"
tokenizer_model = "/data/hf_models/chinese-llama2-13b/tokenizer.model"
# samples_mapping = np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')
dataset = indexed_dataset.MMapIndexedDataset(data_prefix)
tokenizer = _SentencePieceTokenizer(tokenizer_model, 100)

import pdb; pdb.set_trace()
tokenizer.detokenize(dataset[34].tolist())
# f = open('./extra_ids1.txt', 'w')
# for i in range(32007 * 4096, 32008 * 4096):
#     idx = samples_mapping[i]
#     source = list(map(int, dataset[(0, idx)][0]))
#     # target = list(map(int, dataset[(0, idx)][1]))
#     f.write(tokenizer.detokenize(source) + '\n')
#     # f.write(tokenizer.detokenize(target) + '\n')
# f.close()

# data_prefix = "/data/en_zh/val_spancorr"
# tokenizer_model = "/data/tokenizer/multilingual-spiece.model"
# tokenizer = _SentencePieceTokenizer(tokenizer_model, 100)
# dataset = make_dataset(data_prefix, impl='mmap', skip_warmup=True)
# f = open('./extra_ids1.txt', 'w')
# for i in range(len(dataset)):
#     text = list(map(int, dataset[(0, i)][0]))
#     f.write(tokenizer.detokenize(text) + '\n')
# f.close()
