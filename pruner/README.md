# Megatron模型裁剪以及恢复训练 （仅支持llama模型）
[[中文版](README.md)] [[English](README_EN.md)]
## 简介
本方法支持将一个Transformer预训练模型裁剪成一个任意大小的新预训练模型并保留大部分性能。对于一个transformer模型，参数量大小由 ```layer_num, hidden_size, intermediate_size, num_attention_heads```决定。使用此份代码，只需要设置新的``drop_layers,hidden_size_remain,intermediate_size_remain```,```num_attetion_heads_remain``即可将模型裁剪为更小的模型。 

以Llama为例：
| Model | layer_num | hidden_size | intermediate_size | num_attention_heads | 
| :--- | :---: | :---: | :---: | :---: |
| LLaMA-13B | 40 | 5120| 13824 | 40 |
| LLaMA-7B | 32 | 4096| 11006 | 32 | 
| LLaMA-3.4B | 28 | 3072| 8192 | 24 | 
| LLaMA-2.7B | 24 | 2560| 6912 | 20 | 

由于没有对kv_channel进行裁剪，并且llama使用了 SwiGLU-MLP，因此hidden_size与intermediate_size以及num_attention_heads都存在一定比例的对应关系。建议裁剪时先选定hidden_size，然后再计算对应的intermediate_size以及num_attention_heads。

### 深度裁剪
该方法与经典的layerdrop方法相同，通过drop_layers参数可以设定你需要裁剪的层编号。我们进行了一些前期实验，对于固定的裁剪层数，往往从中间裁剪对模型的伤害更小，裁剪的层间隔越远对模型的伤害越小。例如，对于层编号从0开始的12层的transformer如果需要裁剪2层，裁剪4，7两层效果较好。

### 宽度裁剪
设定```hidden_size, intermediate_size, num_attention_heads``` 等参数即可。
该方法将原始模型通过结构化权重裁剪，裁剪成任意大小的模型。例如，对于一个 [n,m] 的矩阵，如果将其裁剪成 [n-a,m-b] 大小的矩阵，将随机裁剪掉 a 行和 b 列的参数。
但实际上, Transformer结构中的权重矩阵有严格的依赖关系。例如，在计算过程中，对于初始向量 x, 会经过多层的网络。例如 $x_1 = x_0AB$ 其中A和B的矩阵相乘存在行列的依赖关系。对于大小为```hidden_size``` 的向量 $x_0$，如果裁剪掉第 i 个位置, 那么 向量A以及向量B分别需要裁剪掉第 i 行以及第 i 列， 这能使得裁剪后矩阵的计算结果在除了 i 位置以外的其他位置保持一致。这份代码会生成随机的裁剪位置，并按照transformer的计算方式进行位置对应的结构化裁剪。

## 使用方法
### 步骤1
将llama的huggingface模型权重转化成megatron模型权重
```
bash tools/checkpoint_conversion/hf_to_megatron.sh
```
可以任意设置的tensor_parallel_size，该方法支持张量并行下的权重裁剪。

### 步骤2
megatron模型裁剪，参考脚本 ```scripts/prune_llama.sh```
```
bash scripts/prune_llama.sh
```
- `--load`: 待裁剪的megatron模型路径。
- `GPT_ARGS`: 所有参数均与原模型参数保持一致即可。
- `PRUNE_ARGS`
    - `--hidden_size_remain`: 裁剪后的 hiddens_size。
    - `--num_attention_heads_remain`: 裁剪后的 attention_heads。
    - `--ffn_hidden_size_remain`:  裁剪后的ffn_hidden_size。
    - `--drop_layers`: 需要裁剪的层数编号，从0开始。
    - `--prune_type`: 默认为 "balance"。由于对于较大的模型是在张量并行下裁剪，要裁剪的行或列可能不在一台GPU上。“balance” 裁剪会让每台GPU上裁剪相同个数的行或者列。即裁剪后，每个子模型大小完全一致。若不使用“balance”则可能每个子模型大小不一致，需要使用transfer.sh脚本转化为tp1的模型。
### 步骤3
裁剪完的模型需要进行少量的恢复训练即能恢复大部分的能力.
模型继续训练的脚本如下：
 ```
 bash scripts/check_model.sh
 ```

## 大模型裁剪实验
我们在目前最流行的llama模型上进行了实验，分别将llama2-13B裁剪至7B以及llama2-7B裁剪至3.5B.

我们在开源的pile数据集对裁剪后模型的恢复训练。
以下是具体的模型参数以及恢复训练后的ml loss：

| Model | layer_num | hidden_size | intermediate_size | num_attention_heads | ml loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| LLaMA2-13B | 40 | 5120| 13824 | 40 | 1.50 |
| LLaMA2-7B | 32 | 4096| 11006 | 32 | 1.54 |
| Pruned-7B | 32 | 4096| 11006 | 32 | 1.56 (20B tokens) |
| Pruned-3.4B | 28 | 3072| 8192 | 28 | 1.71 (12B tokens) | 

### 恢复训练loss如下
#### 不同大小模型裁剪后的loss比较
训练曲线

![avatar](assets/loss.png)

训练表格

| Model | 4B| 8B | 12B |16B | 20B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Pruned-7B | 1.700 | 1.642| 1.616 | 1.587 | 1.553 |
| Pruned-3.4B | 1.839 | 1.746| 1.713 | - | - | 

从训练的loss来看，模型尚未完全收敛，若训练更多的数据的话还有较大的提升空间。
