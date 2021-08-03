这是盘古α模型的 pytorch 实现版本。可以在 pytorch 框架上进行推理、训练、finetune。

出发点：Mindspore 是新的深度学习框架，很多人没用过，所以把 mindspore 模型转成 pytorch 模型可以让更多人使用我们的盘古模型，让使用者不但可以体验我们的大模型，还可以对我们的模型进行 finetune 。

Megatron 是英伟达深度学习应用研究团队开发的一款大型、强大的 transformer 算法库。这次的移植是在 Megatron 的基础上修改得到，主要工作内容包括了模型文件的转换、增加 query layer、修改模型切分策略。

# 配置

支持 python >= 3.6, pytorch >= 1.5, cuda >= 10, and nccl >= 2.6 版本.

推荐使用英伟达的官方 docker 镜像`docker pull nvcr.io/nvidia/pytorch:20.03-py3`。需要安装 [NLTK](https://www.nltk.org/install.html)。

也可直接下载我配好的[镜像文件](https://git.openi.org.cn/attachments/3e743b41-ab0a-4e2a-9bdb-066afc1c8740?type=0) ，`docker load -i ***.tar` 即可，使用`/opt/conda/bin/python`.

# 模型文件下载

| 模型文件                                                     | Md5                              | 大小 | 参数配置                                                     |
| ------------------------------------------------------------ | -------------------------------- | ---- | ------------------------------------------------------------ |
| [Pangu-alpha_2.6B_fp16_mgt.zip](https://git.openi.org.cn/attachments/72aec03d-6bdb-4652-ac2a-8099db4b0bed) | 28f6dd2ec5d1df2fd22ec5f4a66f51e7 | 4.6G | num-layers : 31<br />hidden-size : 2560<br />num-attention-heads : 32 |
| [Pangu-alpha_13B_fp16_mgt.zip](https://git.openi.org.cn/attachments/937b3e2d-98fb-4871-9691-b32afb5a4d79?type=0) | e6f7a05cbdf8ba8d69e6786e48344f6f | 22G | num-layers : 39<br />hidden-size : 5120<br />num-attention-heads : 40 |

注：`num-layers` 等于 Pangu 项目中的 `num-layers - 1`

#精度
模型转换需要先把 mindspore 的 ckpt 转为 numpy 的 npy 文件，然后再把 npy 文件加载到 pytorch 模型。该过程存在精度损失，所以 pytorch 模型的结果和 Pangu-Alpha 的 mindspore 版本的推理结果有一定的差异。暂时还没解决，正在寻找解决方案。

# 推理

目前只有生成文本的推理脚本，如下：

需要配置参数：

`--out-seq-length`：生成的最大 token 数

`--top_k`：k 值越大生成样本多样性越高

```
python tool/generate_samples_Pangu.py \
--model-parallel-size 1 \
--num-layers 31 \
--hidden-size 2560 \
--load /**ckpt path**/ \
--num-attention-heads 32 \
--max-position-embeddings 1024 \
--tokenizer-type GPT2BPETokenizer \
--fp16 \
--batch-size 1 \
--seq-length 1024 \
--out-seq-length 50 \
--temperature 1.0 \
--vocab-file megatron/tokenizer/bpe_4w_pcl/vocab \
--num-samples 0 \
--top_k 10 \
--finetune
```





# Finetune

##### 1、准备训练数据

参考[数据](#数据)部分

##### 2、模型切割

上面下载的模型是单机推理模型，所以在进行 finetune 的时候需要先对模型进行切割，切割成模型并行的模型。

参数：

`model-parallel-size`：原始模型的分片个数，这里是 1

`--num-mp-model`： 切分后的模型个数

`--mp-model-save`：切分后，模型的保存路径

```
python tools/split_full_model_into_mp_model.py \
--model-parallel-size 1 \
--num-mp-model 2 \
--num-layers 31 \
--hidden-size 2560 \
--load /**ful model path**/ \
--mp-model-save /**mp model save path**/ \
--num-attention-heads 32 \
--max-position-embeddings 1024 \
--tokenizer-type GPT2BPETokenizer \
--fp16 \
--batch-size 1 \
--seq-length 1024 \
--model-type Pangu \
--vocab-file megatron/tokenizer/bpe_4w_pcl/vocab \
--finetune
```

##### 3、训练

运行脚本:

```examples/finetune_pangu_distributed.sh```

##### 4、模型合并

finetune 完后的模型是分片的，如果要进行单卡推理，则先需要对模型进行合并。

合并脚本：

`--mp-model-parallel-size`：模型分片数

`--load`：模型保存目录

```
python tool/merge_mp_partitions.py \
--model-parallel-size 1 \
--mp-model-parallel-size 2 \
--num-layers 31 \
--hidden-size 2560 \
--load /full model ckpt dir/  \
--num-attention-heads 32 \
--max-position-embeddings 1024 \
--tokenizer-type GPT2BPETokenizer \
--fp16 \
--batch-size 1 \
--seq-length 1024 \
--model-type Pangu \
--vocab-file megatron/tokenizer/bpe_4w_pcl/vocab \
--finetune \
```



# 训练

参考脚本

```
examples/pretrain_gpt2_distributed_2.6B.sh
```



# 数据

##### 生成训练数据

参考脚本：

`preprocess_data.py`





