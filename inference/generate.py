"""
TopK for text generation
"""

import numpy as np
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
import mindspore as ms


def top_k_logits(logits, top_k=0, top_p=0.9, filter_value=-float(0)):
    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k

        p_args = logits.argsort()[::-1][:top_k]
        mask = np.ones(logits.shape) * filter_value
        mask[p_args] = 1
        logits = logits * mask

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_indices = np.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]

        # cumulative_probs = np.cumsum(softmax(sorted_logits), axis=-1)
        cumulative_probs = np.cumsum(sorted_logits, axis=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def generate(model, origin_inputs, seq_length, end_token=50256):

    pad_length = seq_length - origin_inputs.shape[-1]

    generate_tokens_num = 0
    valid_length = len(origin_inputs)
    label_token_length = 50
    input_ids = np.pad(origin_inputs, ((0, 0), (0, pad_length)), 'constant', constant_values=(0, 0))
    while generate_tokens_num < label_token_length:
        logits = model.predict(ms.Tensor(input_ids, ms.int32)).asnumpy().reshape(1, seq_length, -1)
        probs = top_k_logits(logits[0, valid_length - 1], top_k=5, top_p=0.8)
        p = softmax(probs)
        p = p / sum(p)
        target_index = np.random.choice(len(p), p=p)
        #         print(target_index)

        if (target_index == end_token) or (valid_length == seq_length - 1):
            outputs = input_ids
            break

        if not target_index == 0:  # 后处理，不允许模型生成UNK
            input_ids[0][valid_length] = target_index
            valid_length += 1
            generate_tokens_num += 1
        outputs = input_ids
    outputs = outputs[0, :np.sum(outputs != 0)]
    return outputs
    # print(logits.shape)
    # print(logits)
    # return origin_inputs
