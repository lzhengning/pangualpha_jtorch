# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""

import os
import sys
import numpy as np
import torch
from megatron.text_generation_utils import pad_batch, get_batch
from load_iflytek import iflytek_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPT2Model
from megatron.training import get_model
from megatron.text_generation_utils import generate_and_write_samples_unconditional
from megatron.text_generation_utils import generate_samples_input_from_file
from megatron.text_generation_utils import generate_samples_interactive

# from megatron.model.transformer import LayerNorm
from megatron import mpu

def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_tokentypes=0, parallel_output=False)

    return model


def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=5,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--sample-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                            'each line is an input.')
    group.add_argument("--sample-output-file", type=str, default=None,
                       help='Output file got from --sample-input-file')
    group.add_argument("--num-samples", type=int, default=0,
                       help='Number of samples to generate unconditionally, '
                            'defaults to 0 and interactive conditional sampling')
    group.add_argument("--genfile", type=str,
                       help='Output file when generating unconditionally')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                            'instead of using previously computed keys/values.')

    return parser


def generate(model, context_tokens, args, tokenizer, max_num=50):

    valid_length = len(context_tokens)
    context_tokens_, context_lengths = pad_batch([context_tokens],
                                                 tokenizer.pad_id, args)
    context_tokens_tensor = torch.cuda.LongTensor(context_tokens_)
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)
    type_ids = None
    bs,_  = tokens.shape
    cnt = 0
    while valid_length < args.seq_length:
        with torch.no_grad():
            logits = model(tokens,
                           position_ids,
                           attention_mask,
                           tokentype_ids=type_ids,
                           forward_method_parallel_output=False)
        logits = logits[:,:,:tokenizer.vocab_size].cpu().numpy()
        logits = logits.reshape(bs, args.seq_length, -1)
        probs = logits[0, valid_length-1, :]
        p_args = probs.argsort()[::-1][:args.top_k]

        p = probs[p_args]
        p = p / sum(p)
        for i in range(1000):
            target_index = np.random.choice(len(p), p=p)
            if p_args[target_index] != tokenizer.unk:
                break

        if p_args[target_index] == tokenizer.eod or \
                valid_length == args.seq_length-1 or cnt>=max_num:
            outputs = tokens.cpu().numpy()
            break
        tokens[0][valid_length] = p_args[target_index]
        valid_length += 1
        cnt += 1

    length = np.sum(outputs != tokenizer.pad_id)
    outputs = outputs[0][:length]
    return outputs


def main():
    """Main program."""

    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

    # Set up model and load checkpoint.
    model = get_model(model_provider)
    model.eval()

    args = get_args()
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    task = 'zero_shot'
    dataset = iflytek_dataset()
    samples = dataset.getSamples()

    acc = []
    for ind,sample in enumerate(samples):
        loss_ = []
        for class_sample,loss_mask in zip(sample['input_ids_list'],sample['loss_mask_list']):
            context_tokens_tensor = torch.cuda.LongTensor([class_sample])
            labels = context_tokens_tensor[:, 1:].contiguous().cuda()
            tokens = context_tokens_tensor[:, :-1].contiguous()
            tokens, attention_mask, position_ids = get_batch(tokens)
            type_ids = None
            with torch.no_grad():
                logits = model(tokens,
                               position_ids,
                               attention_mask,
                               tokentype_ids=type_ids,
                               forward_method_parallel_output=False)
            losses = mpu.vocab_parallel_cross_entropy(
                logits.contiguous().float(), labels.contiguous())
            loss_mask = torch.cuda.LongTensor(loss_mask)
            loss = torch.sum(
                losses.view(-1) * loss_mask.contiguous().view(-1).float())
            loss_.append(loss)

        predict = np.argmin(loss_)
        acc.append(predict==sample['label'])
        if (ind%100 == 0):
            print(f'[{ind}] iflytek-{task}: acc={sum(acc)}/{len(acc)}={sum(acc)/len(acc)}')

    accuracy = sum(acc)/len(acc)
    pass


if __name__ == "__main__":

    main()
