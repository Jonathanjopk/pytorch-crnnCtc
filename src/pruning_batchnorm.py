# coding=UTF-8
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from model import CRNN


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming yolov4 prune')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--img_channel', type=int, default=32,
                    help='image channel at first (default: 32)')
parser.add_argument('--img_height', type=int, default=128,
                    help='image height (default: 128)')
parser.add_argument('--img_width', type=int, default=128,
                    help='image width (default: 128)')
parser.add_argument('--num_class', type=int, default=5,
                    help='number of class (default: 5)')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
# parser.add_argument('--model', default='', type=str, metavar='PATH',
#                     help='path to the model (default: none)') # 回研發雲後要用
parser.add_argument('--save', default='./', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

if args.cuda:
    model.cuda()

# 回研發雲後用checkpoint
# if args.model:
#     if os.path.isfile(args.model):
#         print("=> loading checkpoint '{}'".format(args.model))
#         checkpoint = torch.load(args.model) # model = torch.load('./weights/yolov4.pt')
#         args.start_epoch = checkpoint['epoch']
#         best_prec1 = checkpoint['best_prec1']
#         model.load_state_dict(checkpoint['state_dict'])
#         print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
#               .format(args.model, checkpoint['epoch'], best_prec1))
#     else:
#         print("=> no checkpoint found at '{}'".format(args.resume))


def compute_thre(crnn_module):
    """計算global threshold"""
    # 計算總共多少channels
    total = 0
    for m in crnn_module.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0] # m.weight就是gamma

    # 所有gamma值 取絕對值存進bn
    bn = torch.zeros(total) # 1*n維
    index = 0
    for m in crnn_module.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0] # channels
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size
    # 由小到大排序
    y, i = torch.sort(bn) # 小 -> 大
    thre_index = int(total * args.percent) # scale sparse rate 0.5 剪枝比例
    thre = y[thre_index] if thre_index != 0 else 0 # 取第thre_index個值當作threshold，如果 thre_index=0 代表全留，不能取第 0 個要直接改 0
    # 之後weight會跟thre這個數字比大小，產生一個0, 1的tensor，大於thre的留下(小於thre的就不會被存進newmodel)
    return thre, total


def prune_channels(model_key, thre, skip, total):
    """記錄誰該留下誰該剪掉"""
    pruned = 0
    # 要先有第一層 image channel
    cfg_new = [args.img_channel] # remaining channel
    cfg_mask = [torch.ones(args.img_channel)] # 記錄每層channels，以0,1表示剪枝，假設channels=3, cfg_mask=[0,1,1]
    for k, m in enumerate(model_key.modules()):
        if isinstance(m, nn.BatchNorm2d):
            thre_ = 0 if k in skip else thre # skip的layer thre=0
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre_).float() # 比大小，大的標記1&小的標記0，存進mask

            cfg_new.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())

            pruned = pruned + mask.shape[0] - torch.sum(mask) # 計算pruning ratio
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(k, mask.shape[0], int(torch.sum(mask))))
    pruned_ratio = pruned / total
    # print(pruned, total)
    print('pruned ratio:', pruned_ratio)
    return cfg_mask, cfg_new

def get_new_weights(old_model, new_model, cfg_dict):
    old_modules = list(old_model.modules())
    new_modules = list(new_model.modules())
    layer_id_in_cfg = 0
    cfg_mask = cfg_dict['cfg_mask']
    cat_layer = cfg_dict['cat_layer']
    start_mask = cfg_mask[layer_id_in_cfg]
    end_mask = cfg_mask[layer_id_in_cfg+1]

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]

        # 針對 conv
        if isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('=====================================================')
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone() # in_channel
            w1 = w1[idx1.tolist(), :, :, :].clone() # out_channel
            m1.weight.data = w1.clone() # 存入新的權重

        # 針對 batchnorm
        elif isinstance(m0, nn.BatchNorm2d):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            # 存入新的權重
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            
            # 跑最後一層會有 list index 超出範圍，所以限制
            if layer_id_in_cfg < 6:
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                end_mask = cfg_mask[layer_id_in_cfg+1]
    return new_modules

def main(model):
    # 新的 cfg
    '''
    model: 各架構位置
    skip: 不剪枝的層數
    cfg: 剪枝後剩餘的 channel 數量
    cfg_mask: 剪枝後剩餘 channel 的位置
    cat_layer: 有 concat 的層數
    '''
    pruning_cfg = {
        'cnn':{
            'model': model.cnn,
            'skip': [],
            'cfg': [],
            'cfg_mask': [],
            'cat_layer': []
        }
    }

    # 進行剪枝
    for model_key in pruning_cfg.keys():
        # 取threshold, total channels
        thre, total = compute_thre(pruning_cfg[model_key]['model'])
        print(thre, total)
        # 取剪完的cfg_mask, cfg_new
        cfg_mask, cfg_new = prune_channels(pruning_cfg[model_key]['model'],
                                           thre,
                                           pruning_cfg[model_key]['skip'],
                                           total)
        # 存進pruning_cfg裡
        pruning_cfg[model_key]['cfg'] = cfg_new
        pruning_cfg[model_key]['cfg_mask'] = cfg_mask
        print('Pre-processing Successful!')

    # 用新的pruning_cfg定義新模型架構
    newmodel = CRNN(img_channel=args.img_channel, 
                    img_height=args.img_height, 
                    img_width=args.img_width, 
                    num_class=args.num_class, 
                    pruning_cfg=pruning_cfg['cnn']['cfg'])
    newmodel_cfg = {
        'cnn': {'model': newmodel.cnn}
    }

    # 存入新的權重
    key_list = list(pruning_cfg.keys())
    for key_id in range(len(key_list)):
        print(f'model_key: {key_list[key_id]}')
        new_crnn_module = get_new_weights(old_model=pruning_cfg[key_list[key_id]]['model'],
                                          new_model=newmodel_cfg[key_list[key_id]]['model'],
                                          cfg_dict=pruning_cfg[key_list[key_id]])
        # 將 model weights 存進新的 new_model 裡
        newmodel_cfg[key_list[key_id]]['model'] = new_crnn_module[0]

    print('Successful all pruning process!!!')
    # torch.save(newmodel, './weights/test_newmodel.pth')
    # torch.save({'cfg': pruning_cfg[model_key]['cfg'], 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))


if __name__ == '__main__':
    model = CRNN(img_channel=args.img_channel, img_height=args.img_height, img_width=args.img_width, num_class=args.num_class)
    # model = torch.load('crnn.pt')
    main(model)


