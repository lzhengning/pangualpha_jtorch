# PanGu-Alpha-GPU

 

### 描述

本项目是  [Pangu-alpha](https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha) 的 GPU 推理版本，详情请查看原项目。目前可以正确在 GPU 单卡上推理，正在尝试训练流程。



### 方法

目前个人只成功在 T5 GPU 上运行2.6B，由于显存有限，所以把模型参数压缩成 `fp16` 。13B 模型的推理也正在做。

##### 环境

```
yands/mindspore_pangu-alpha:1.2.0
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



gpu 使用情况

| 模型      | 显存占用 |
| --------- | -------- |
| 2.6B_fp16 | 6728.0M  |



### MD5

`Pangu-alpha_2.6B.ckpt`的 md5 是：`da404a985671f1b5ad913631a4e52219 `



### 问题

目前可以在 T5 GPU上正确推理，尝试在 v100 上运行，但报错

```
log:
Exception in thread Thread-3:
Traceback (most recent call last):
  File "/usr/local/python-3.7.5/lib/python3.7/threading.py", line 926, in _bootstrap_inner
    self.run()
  File "/usr/local/python-3.7.5/lib/python3.7/threading.py", line 870, in run
    self._target(*self._args, **self._kwargs)
  File "/usr/local/python-3.7.5/lib/python3.7/multiprocessing/pool.py", line 470, in _handle_results
    task = get()
  File "/usr/local/python-3.7.5/lib/python3.7/multiprocessing/connection.py", line 251, in recv
    return _ForkingPickler.loads(buf.getbuffer())
ModuleNotFoundError: No module named 'tvm'

Traceback (most recent call last):
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/_extends/parallel_compile/akg_compiler/akg_process.py", line 128, in compile
    res.get(timeout=self.wait_time)
  File "/usr/local/python-3.7.5/lib/python3.7/multiprocessing/pool.py", line 653, in get
    raise TimeoutError
multiprocessing.context.TimeoutError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/_extends/remote/kernel_build_server_gpu.py", line 87, in <module>
    messager.run()
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/_extends/remote/kernel_build_server.py", line 119, in run
    self.loop()
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/_extends/remote/kernel_build_server.py", line 116, in loop
    self.handle()
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/_extends/remote/kernel_build_server_gpu.py", line 56, in handle
    res = self.akg_builder.compile()
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/_extends/remote/kernel_build_server.py", line 33, in compile
    return self.akg_builder.compile()
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/_extends/parallel_compile/akg_compiler/akg_process.py", line 128, in compile
    res.get(timeout=self.wait_time)
  File "/usr/local/python-3.7.5/lib/python3.7/multiprocessing/pool.py", line 623, in __exit__
    self.terminate()
  File "/usr/local/python-3.7.5/lib/python3.7/multiprocessing/pool.py", line 548, in terminate
    self._terminate()
  File "/usr/local/python-3.7.5/lib/python3.7/multiprocessing/util.py", line 201, in __call__
    res = self._callback(*self._args, **self._kwargs)
  File "/usr/local/python-3.7.5/lib/python3.7/multiprocessing/pool.py", line 585, in _terminate_pool
    "Cannot have cache with result_hander not alive")
AssertionError: Cannot have cache with result_hander not alive
[ERROR] SESSION(3269,python):2021-05-12-09:48:57.778.803 [mindspore/ccsrc/backend/session/kernel_build_client.h:110] Response] Response is empty
Traceback (most recent call last):
  File "integrate_checkpoint.py", line 217, in <module>
    integrate_ckpt_file()
  File "integrate_checkpoint.py", line 141, in integrate_ckpt_file
    output_ids = generate(model_predict, input_ids, config.seq_length, 9)
  File "/userhome/pclproject/gpt/transformModelToGPU/generate.py", line 46, in generate
    logits = model.predict(ms.Tensor(input_ids, ms.int32)).asnumpy().reshape(1, seq_length, -1)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/train/model.py", line 791, in predict
    result = self._predict_network(*predict_data)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/nn/cell.py", line 341, in __call__
    out = self.compile_and_run(*inputs)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/nn/cell.py", line 608, in compile_and_run
    self.compile(*inputs)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/nn/cell.py", line 595, in compile
    _executor.compile(self, *inputs, phase=self.phase, auto_parallel_mode=self._auto_parallel_mode)
  File "/usr/local/python-3.7.5/lib/python3.7/site-packages/mindspore/common/api.py", line 494, in compile
    result = self._executor.compile(obj, args_list, phase, use_vm)
RuntimeError: mindspore/ccsrc/backend/session/kernel_build_client.h:110 Response] Response is empty
```