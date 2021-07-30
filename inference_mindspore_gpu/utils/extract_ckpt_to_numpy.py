
from mindspore import save_checkpoint, load_checkpoint, build_searched_strategy, merge_sliced_parameter
from mindspore import ops
import numpy as np

from mindspore import context

if __name__ == '__main__':

    context.set_context(mode=context.PYNATIVE_MODE)

    pwd = '/ghome/yands/model/'
    file_name = pwd + 'PanguAlpha_2.6B_fp16.ckpt'
    # file_name = '/Users/sam/Downloads/PanguAlpha_2.6B_fp16.ckpt'
    param_dict1 = load_checkpoint(file_name)

    keys_same = []
    for key in param_dict1.keys():
        equal_count = ops.EqualCount()
        a = param_dict1[key]
        parameter_name = a.name
        parameter_shape = a.data.shape
        parameter_shape_length = len(parameter_shape)
        print(type(param_dict1[key]))
        print(param_dict1[key])

    pass