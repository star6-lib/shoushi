import torch
import argparse
import yaml

from lib.model_plate import GANO
from lib.utils_plate_train import train, sup_train, plus_train
from lib.utils_data import generate_plate_stress_data_loader

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# define arguements
parser = argparse.ArgumentParser(description='command setting')
parser.add_argument('--phase', type=str, default='plus_train')
parser.add_argument('--data', type=str, default='plate_stress_DG')
parser.add_argument('--model', type=str, default='GANO')
parser.add_argument('--geo_node', type=str, default='vary_bound', choices=['vary_bound', 'all_bound', 'all_domain'])

args = parser.parse_args()

# extract configuration
with open(r'./configs/{}_{}.yaml'.format(args.model, args.data), 'r') as stream:
    config = yaml.load(stream, yaml.FullLoader)

# define device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define model
# if args.model == 'DCON':
#     model = DCON(config)
if args.model == 'GANO':
    model = GANO(config)
# if args.model == 'self_defined':
#     model = New_model_plate(config)

# ================= 加入这行保护锁 =================
if __name__ == '__main__':
    print('Model forward phase: {}'.format(args.phase))
    print('Using dataset: {}'.format(args.data))
    print('Using model: {}'.format(args.model))
    # 必须把读取数据和执行训练的代码缩进到这个 if 里面！
    train_loader, val_loader, test_loader, num_nodes_list, params = generate_plate_stress_data_loader(args, config)

    # 根据命令行传入的 phase 参数，动态选择训练模式
    if args.phase == 'train':
        print("\n>>> 启动 [物理信息驱动 PINN] 训练模式...\n")
        train(args, config, model, device, (train_loader, val_loader, test_loader), num_nodes_list, params)

    elif args.phase == 'sup_train':
        print("\n>>> 启动 [纯数据驱动 Data-Driven] 训练模式...\n")
        sup_train(args, config, model, device, (train_loader, val_loader, test_loader), num_nodes_list, params)
    elif args.phase == 'plus_train':
        print("\n>>> 启动 [物理-数据双驱动 Dual-Driven] 训练模式...\n")
        plus_train(args, config, model, device, (train_loader, val_loader, test_loader), num_nodes_list, params)
    else:
        print("\n>>> 未知模式，请检查 --phase 参数设定！")



