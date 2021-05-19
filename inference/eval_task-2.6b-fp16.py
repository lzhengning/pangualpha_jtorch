from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
import mindspore as ms
from pangu_dropout_recompute_eos_fp16 import PANGUALPHA,EvalNet
from utils_fix import PANGUALPHAConfig

from tokenization_jieba import JIEBATokenizer
from generate import generate
import os
import numpy as np


from gpt_dropout_recompute_eos_fp16 import EvalNet, GPT, EvalNet_p
from inference.gpt_wrapcell_gradient_scale_eos import VirtualDatasetOneInputCell


def get_model():

    # model_parallel_num = 8
    model_parallel_num = 1
    data_parallel_num = int(1 / model_parallel_num)
    per_batch_size = 1
    batch_size = per_batch_size * data_parallel_num
    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=1024,
        vocab_size=40000,
        embedding_size=5120,  # 5120,  # 353M   8B
        num_layers=40,  # 40,
        num_heads=40,  # ,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,  # 0.0,
        compute_dtype=ms.float16,
        use_past=False,
        self_layernorm=True,
        forward_reduce_scatter=True,
        word_emb_dp=True,
        eod_reset=False,
        word_emb_path='Newexp69_GPT3_5-684_2_word_embedding.npy',
        position_emb_path='Newexp69_GPT3_5-684_2_position_embedding.npy',
        top_query_path='Newexp69_GPT3_5-684_2_top_query_embedding.npy')
    print("===config is: ", config, flush=True)
    gpt = GPT(config)
    gpt_ = VirtualDatasetOneInputCell(gpt)
    eval_gpt = EvalNet_p(gpt_, generate=True)
    eval_gpt.set_train(False)

    model = Model(eval_gpt)


    param_dict = load_checkpoint('/userhome/temp/PanguAlpha_13b_fp16.ckpt')
    load_param_into_net(eval_gpt, param_dict)

    print('#### Load ckpt success!!! ####')

    return model

def get_model_pangu():

    eod_reset = False
    model_parallel_num = 1
    data_parallel_num = int(1 / model_parallel_num)
    per_batch_size = 1
    batch_size = per_batch_size * data_parallel_num

    config = PANGUALPHAConfig(
        data_parallel_num=data_parallel_num,
        model_parallel_num=model_parallel_num,
        batch_size=batch_size,
        seq_length=1024,
        vocab_size=40000,
        embedding_size=2560,  # 353M   8B
        num_layers=32,
        num_heads=32,
        expand_ratio=4,
        post_layernorm_residual=False,
        dropout_rate=0.1,
        compute_dtype=ms.float16,
        use_past=False,
        self_layernorm=True,
        forward_reduce_scatter=True,
        word_emb_dp=True,
        eod_reset=eod_reset)

    pangualpha = PANGUALPHA(config)
    print('initial PANGU-ALPHA success!!!')

    eval_net = EvalNet(pangualpha)
    eval_net.set_train(False)
    model_predict = Model(eval_net)

    param_dict = load_checkpoint('/userhome/temp/2.6b_ckpts/new.ckpt')
    load_param_into_net(eval_net, param_dict)

    print('load_param_into_net success!!!!!!!!')
    print("================load param ok=================", flush=True)


def run_eval():

    ms.context.set_context(save_graphs=False, mode=ms.context.GRAPH_MODE, device_target="GPU")

    model_predict = get_model()

    tokenizer_path = os.getcwd() + "/tokenizer"
    tokenizer = JIEBATokenizer(os.path.join(tokenizer_path, 'vocab.vocab'),
                               os.path.join(tokenizer_path, 'vocab.model'))

    while 1:
        sample = input("Tell Pangu-alpha what you want to generate:")
        tokenized_token = tokenizer.tokenize(sample)
        start_sentence = tokenizer.convert_tokens_to_ids(tokenized_token)
        input_ids = np.array(start_sentence).reshape(1, -1)
        output_ids = generate(model_predict, input_ids, 1024, 9)
        output_samples = tokenizer.convert_ids_to_tokens(output_ids.tolist())
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('Input is:', sample)
        print('Output is:', output_samples, flush=True)
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    return

if __name__ == "__main__":
    run_eval()


