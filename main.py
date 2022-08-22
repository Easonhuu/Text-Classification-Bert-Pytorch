# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import os
import argparse
from utils import build_dataset, build_iterator, get_time_dif


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) 

    model_name = 'bert'  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset, current_time)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)
    log_path = os.path.join(config.save_path, 'log.txt')
    config.f_log = open(log_path, 'w')

    start_time = time.time()
    print("Loading data...")
    print("Loading data...", file=config.f_log)
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage: %s" % time_dif)
    print("Time usage: %s" % time_dif, file=config.f_log)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
    config.f_log.close()
