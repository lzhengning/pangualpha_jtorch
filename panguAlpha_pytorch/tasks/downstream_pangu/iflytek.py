import os
import json
import time
import itertools
import numpy as np

zc_cache = []

def load_iflytek_train_example_for_shot(data_path, num_sample=2, np_rng=None, max_len=None, input_str_format=None):
    if input_str_format is None:
        input_str_format = "这是关于{label}的应用程序：{sentence}"
    # input_str_format = "{s}：{label}"
    if np_rng is None:
        np_rng = np.random.default_rng()
    if len(zc_cache)>0:
        z0 = zc_cache[0]
    else:
        tmp0 = [os.path.join(data_path,x) for x in os.listdir(data_path) if x.startswith('train')]
        assert len(tmp0)==1
        train_file = tmp0[0]
        with open(train_file, 'r') as fid:
            z0 = [json.loads(x) for x in fid.readlines()]
        zc_cache.append(z0)

    # select sample with balanced labels
    hf0 = lambda x: x[1]
    tmp0 = itertools.groupby(sorted([(x,y['label_des']) for x,y in enumerate(z0)],key=hf0), key=hf0)
    group_index = [np.array([z[0] for z in y]) for x,y in tmp0]
    for x in group_index:
        np_rng.shuffle(x) #in-place
    tmp0 = (num_sample-1)//len(group_index) + 1
    tmp1 = np.concatenate([x[:tmp0] for x in group_index])
    np_rng.shuffle(tmp1)
    selected_index = tmp1[:num_sample]
    # selected_index = np_rng.permutation(len(z0))[:num_sample]

    examples = []
    for x in selected_index:
        sentence = z0[x]['sentence'] if max_len is None else z0[x]['sentence'][:max_len]
        tmp0 = input_str_format.format(label=z0[x]['label_des'], sentence=sentence)
        examples.append(tmp0)
    ret = {
        'zero_shot': '',
        'one_shot': examples[0]+'\n',
        'few_shot':('\n'.join(examples)) + '\n',
    }
    return ret

if __name__ == "__main__":
    # 2.6B-3W
    # word_embedding_path = 's3://mindspore-file/huangxinjing/filtered_ckpt/Newexp65_GPT3-16000_2/Newexp65_GPT3-16000_2_word_embedding.npy'
    # position_embedding_path = 's3://mindspore-file/huangxinjing/filtered_ckpt/Newexp65_GPT3-16000_2/Newexp65_GPT3-16000_2_position_embedding.npy'
    # top_query_embedding_path = 's3://mindspore-file/huangxinjing/filtered_ckpt/Newexp65_GPT3-16000_2/Newexp65_GPT3-16000_2_top_query_embedding.npy'
    # ckpt_path_obs = 's3://mindspore-file/huangxinjing/filtered_ckpt/Newexp65_GPT3-16000_2/Newexp65_GPT3-16000_2part'

    # 2.6B-7W
    word_embedding_path = 's3://mindspore-file/huangxinjing/filtered_ckpt/Newexp65_GPT3_2-3494_2/Newexp65_GPT3_2-3494_2_word_embedding.npy'
    position_embedding_path = 's3://mindspore-file/huangxinjing/filtered_ckpt/Newexp65_GPT3_2-3494_2/Newexp65_GPT3_2-3494_2_position_embedding.npy'
    top_query_embedding_path = 's3://mindspore-file/huangxinjing/filtered_ckpt/Newexp65_GPT3_2-3494_2/Newexp65_GPT3_2-3494_2_top_query_embedding.npy'
    ckpt_path_obs = 's3://mindspore-file/huangxinjing/filtered_ckpt/Newexp65_GPT3_2-3494_2/Newexp65_GPT3_2-3494_2part'

    strategy_ckpt_path_obs = 's3://mindspore-file/strategy_ckpt/gpt_1024_13b_exp65cktp_strategy.ckpt'
    strategy_ckpt_path = '/cache/gpt_1024_13b_exp65cktp_strategy.ckpt'

    rank_id = int(os.environ['RANK_ID'])
    device_id = int(os.environ['DEVICE_ID'])
    world_size = int(os.environ['RANK_SIZE'])
    ckpt_path = '/cache/ckpt_file'
    data_path = hf_project('data/iflytek_public')
    full_batch = True
    eod_reset = False
    is_distribute = True

    sync_file = hf_project('install.txt')
    if rank_id % 8 == 0:
        os.system('ulimit -s 102400')
        mox.file.copy(word_embedding_path, '/cache/word_embedding.npy')
        mox.file.copy(position_embedding_path, '/cache/position_embedding.npy')
        mox.file.copy(top_query_embedding_path, '/cache/top_query_embedding.npy')
        mox.file.copy(strategy_ckpt_path_obs, strategy_ckpt_path)
        for x in ['_0.tar', '_1.tar', '_2.tar', '_3.tar']:
            mox.file.copy(ckpt_path_obs+x, os.path.join(ckpt_path, 'model.tar'))
            os.system(f'cd {ckpt_path}; tar -xf model.tar')
        f = open(sync_file, 'w')
        f.close()
    while not os.path.exists(sync_file):
        time.sleep(1)
    ms.context.set_context(save_graphs=False, mode=ms.context.GRAPH_MODE, device_target="Ascend",
                           device_id=device_id, variable_memory_max_size="30GB")
    if is_distribute:
        ms.communication.management.init()
        ms.context.reset_auto_parallel_context()
        ms.context.set_auto_parallel_context(
            parallel_mode=ms.context.ParallelMode.SEMI_AUTO_PARALLEL,
            gradients_mean=False,
            device_num=world_size,
            full_batch=full_batch,
            strategy_ckpt_load_file=strategy_ckpt_path,
            enable_parallel_optimizer=False)
        auto_parallel_context().set_loss_repeated_mean(True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        _set_multi_subgraphs()

    model_parallel_num = 8
    data_parallel_num = int(world_size / model_parallel_num)
    per_batch_size = 1
    batch_size = per_batch_size * data_parallel_num
    config = GPTConfig(
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
    gpt = GPT(config)
    eval_gpt = GPTWithLoss2(config, gpt, CrossEntropyLoss(config), eos_token=9)
    eval_gpt = VirtualDatasetOneInputCell(eval_gpt)
    eval_gpt.set_train(False)
    model = ms.train.Model(eval_gpt)
    fake_input = ms.Tensor(np.ones(shape=(1, config.seq_length + 1)), ms.int32)
    mask_ids = ms.Tensor(np.ones(shape=(1, config.seq_length)), ms.float32)
    predict_layout = model.infer_predict_layout(fake_input, mask_ids)
    ckpt_file_list = [os.path.join(ckpt_path, f'filerted_{x}.ckpt') for x in range(512)]
    load_distributed_checkpoint(eval_gpt, ckpt_file_list, predict_layout)

    tokenizer = JIEBATokenizer(hf_project('bpe_4w_pcl/vocab.vocab'), hf_project('bpe_4w_pcl/vocab.model'))
    tokenizer.tokenize('我么哦烘干机')

    with open(os.path.join(data_path, 'train.json'), "r", encoding="utf-8") as fid:
        ground_truth = [json.loads(x) for x in fid]
    id_to_label = {int(x['label']):x['label_des'] for x in ground_truth}
    assert set(id_to_label.keys())==set(range(len(id_to_label)))
    label_to_id = {v:k for k,v in id_to_label.items()}

    tmp0 = [
        ('task', ['few_shot']), #'zero_shot','one_shot','few_shot'
        ('max_len', [25]), #None,200,100
        ('tag_new_example', [True]), #True, False
        ('few_shot_num_sample', [3]), #2,3,4
        ('np_seed', [233]), #233,235,237,239
        ('new_mask', [False]), #True, False
        ('input_str_format', [
            # "{label}：{sentence}",
            "这是关于{label}的应用程序：{sentence}",
        ])
    ]
    para_config_list = [{y0[0]:y1 for y0,y1 in zip(tmp0,x)} for x in itertools.product(*[x[1] for x in tmp0])]
    # task = 'few_shot' #['zero_shot','one_shot','few_shot']
    for para_config_i in para_config_list:
        if rank_id==0:
            print(para_config_i)
        task = para_config_i['task']
        max_len = para_config_i['max_len']
        tag_new_example = para_config_i['tag_new_example']
        few_shot_num_sample = para_config_i['few_shot_num_sample']
        np_seed = para_config_i['np_seed']
        new_mask = para_config_i['new_mask']
        input_str_format = para_config_i['input_str_format']
        input_str_format_mask = input_str_format.rsplit('{',1)[0]
        input_str_format_mask_tag_label = '{label}' in input_str_format_mask
        np_rng = np.random.default_rng(seed=np_seed) #must be same across model-parallel

        with open(os.path.join(data_path, 'dev.json'), "r", encoding="utf-8") as fid:
            tmp0 = [json.loads(x) for x in fid] #[:200]
            ground_truth = [tmp0[x] for x in np_rng.permutation(len(tmp0))]

        z0 = []
        zc_print_ind = 0
        if not tag_new_example:
            shot_to_example = load_iflytek_train_example_for_shot(data_path, num_sample=few_shot_num_sample,
                                                                  np_rng=np_rng, max_len=max_len, input_str_format=input_str_format)
            example = shot_to_example[task]
        for instance in ground_truth:
            zc_print_ind += 1
            if tag_new_example:
                shot_to_example = load_iflytek_train_example_for_shot(data_path, num_sample=few_shot_num_sample,
                                                                      np_rng=np_rng, max_len=max_len, input_str_format=input_str_format)
                example = shot_to_example[task]

            true_label = instance['label_des']
            tmp0 = sorted(list(set(id_to_label.values()) - {true_label}))
            fake_label = [tmp0[x] for x in np_rng.permutation(len(tmp0))[:3]] #[:119]
            instance_tf_label = [true_label] + fake_label
            instance_tf_label = [instance_tf_label[x] for x in np_rng.permutation(len(instance_tf_label))] #shuffle
            input_ids_list = []
            mask_list = []
            label_list = []
            input_str_list = []
            for label_i in instance_tf_label:
                if new_mask:
                    tmp0 = tokenizer.tokenize(example)
                else:
                    if input_str_format_mask_tag_label:
                        tmp0 = example + input_str_format_mask.format(label=label_i)
                    else:
                        tmp0 = example + input_str_format_mask.format(sentence=instance['sentence'])
                    tmp0 = tokenizer.tokenize(tmp0)
                tmp1 = example + input_str_format.format(label=label_i, sentence=instance['sentence'])
                input_ids = tokenizer.tokenize(tmp1)[:config.seq_length]
                input_str_list.append(tmp1)

                # tmp0 = tokenizer.tokenize(f"{example}{instance['sentence']}")
                # input_ids = tokenizer.tokenize(f"{example}{instance['sentence']}：{label_i}")[:config.seq_length]

                mask = np.zeros(config.seq_length)
                mask[len(tmp0):len(input_ids)] = 1
                # mask[:len(input_ids)] = 1
                input_ids = np.pad(input_ids, ((0,config.seq_length+1-len(input_ids)),), 'constant', constant_values=(0,9))
                input_ids_list.append(input_ids)
                mask_list.append(mask)
                label_list.append(label_i)
            if (zc_print_ind < 3) and (rank_id==0):
                print(input_str_list[-1]+'\n')

            tmp0 = [ms.Tensor(x[np.newaxis],dtype=ms.int32) for x in input_ids_list]
            tmp1 = [ms.Tensor(x[np.newaxis],dtype=ms.float32) for x in mask_list]
            label = [x for x,y in enumerate(label_list) if y==true_label][0]
            z0.append((tmp0,tmp1,label,input_str_list))

        cnt = 0
        correct_num = 0
        for ind0,(input_ids_list,mask_list,label,input_str_list) in enumerate(z0):
            cnt += 1
            loss = np.concatenate([model.predict(x, y).asnumpy() for x,y in zip(input_ids_list,mask_list)])
            if np.argmin(loss)==label:
                correct_num += 1

            # if rank_id==0:
            #     for ind1,(input_str_i,loss_i) in enumerate(zip(input_str_list, loss)):
            #         print(f'[zcdebug][dev-{ind0}]', input_str_i, ind1==label, loss_i)

            if (ind0%100 == 0) and (rank_id==0):
                print(f'[{ind0}/{len(ground_truth)}] iflytek-{task}: acc={correct_num}/{cnt}={correct_num/cnt}')
        if rank_id==0:
            print(f'[zc-info][2.6B] ckpt={ckpt_path_obs}')
            print(f'{para_config_i} iflytek-{task}: acc={correct_num}/{cnt}={correct_num/cnt}')
