# PanGu-Alpha-GPU

 

### 描述

本项目是  [Pangu-alpha](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha) 的 GPU 推理版本，详情请查看原项目。目前可以正确在 GPU 单卡上推理，正在尝试训练流程。



### 方法

目前个人只成功在 T4 GPU 上运行2.6B，由于显存有限，所以把模型参数压缩成 `fp16` 。13B 模型的推理也正在做。

##### 环境

```
docker pull yands/mindspore_pangu-alpha:1.2.0
```

##### 依赖

```
pip install jieba
pip install sentencepiece==0.1.94
```

##### 运行

```
python eval_task-2.6b-fp16.py
```

##### 结果

由于对模型进行了压缩所以效果可能会比`fp32`差点



**gpu 使用情况**

| 模型      | 显存占用 |
| --------- | -------- |
| 2.6B_fp16 | 6728.0M  |



### 模型

| 模型                                                         | MD5                              |
| ------------------------------------------------------------ | -------------------------------- |
| [Pangu-alpha_2.6B.ckpt](https://git.openi.org.cn/attachments/27234961-4d2c-463b-9052-0240cc7ff29b?type=0) | da404a985671f1b5ad913631a4e52219 |


