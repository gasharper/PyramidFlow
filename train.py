import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import time, argparse
from sklearn.metrics import roc_auc_score

from model import PyramidFlow
from util import MVTecAD, BatchDiffLoss
from util import fix_randseed, compute_pro_score_fast, getLogger

def train(logger, save_name, cls_name, datapath, resnetX, num_layer, vn_dims, \
          ksize, channel, num_stack, device, batch_size, save_memory, ):
    # save config
    save_dict = {'cls_name': cls_name, 'resnetX': resnetX, 'num_layer': num_layer, 'vn_dims': vn_dims,\
                 'ksize': ksize, 'channel': channel, 'num_stack': num_stack, 'batch_size': batch_size}
    loader_dict = fix_randseed(seed=0)

    # model 
    flow = PyramidFlow(resnetX, channel, num_layer, numStack, ksize, vn_dims, saveMem).to(device)
    x_size = 256 if resnetX==0 else 1024
    optimizer = torch.optim.Adam(flow.parameters(), lr=2e-4, eps=1e-04, weight_decay=1e-5, betas=(0.5, 0.9)) # using cs-flow optimizer
    Loss = BatchDiffLoss(batch_size, p=2)

    # dataset
    train_dataset = MVTecAD(cls_name, mode='train', x_size=x_size, y_size=256, datapath=datapath)
    val_dataset = MVTecAD(cls_name, mode='val', x_size=x_size, y_size=256, datapath=datapath)
    test_dataset = MVTecAD(cls_name, mode='test', x_size=x_size, y_size=256, datapath=datapath)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, persistent_workers=True, pin_memory=True, drop_last=True, **loader_dict)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True, drop_last=False, **loader_dict)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1, persistent_workers=True, pin_memory=True, **loader_dict)

    # training & evaluation
    pixel_auroc_lst = [0]
    pixel_pro_lst = [0]
    image_auroc_lst = [0]
    losses_lst = [0]
    t0 = time.time()
    for epoch in range(15):
        # train
        flow.train()
        losses = []
        for train_dict in train_loader:
            image, labels = train_dict['images'].to(device), train_dict['labels'].to(device)
            optimizer.zero_grad()
            pyramid2= flow(image)
            diffes = Loss(pyramid2)
            diff_pixel = flow.pyramid.compose_pyramid(diffes).mean(1)  
            loss = torch.fft.fft2(diff_pixel).abs().mean() # Fourier loss
            loss.backward()
            nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1e0) # Avoiding numerical explosions
            optimizer.step()
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        logger.info(f'Epoch: {epoch}, mean_loss: {mean_loss:.4f}, time: {time.time()-t0:.1f}s')
        losses_lst.append(mean_loss)
        
        # val for template 
        flow.eval()
        feat_sum, cnt = [0 for _ in range(num_layer)], 0
        for val_dict in val_loader:
            image = val_dict['images'].to(device)
            with torch.no_grad():
                pyramid2= flow(image) 
                cnt += 1
            feat_sum = [p0+p for p0, p in zip(feat_sum, pyramid2)]
        feat_mean = [p/cnt for p in feat_sum]

        # test
        flow.eval()
        diff_list, labels_list = [], []
        for test_dict in test_loader:
            image, labels = test_dict['images'].to(device), test_dict['labels']
            with torch.no_grad():
                pyramid2 = flow(image) 
                pyramid_diff = [(feat2 - template).abs() for feat2, template in zip(pyramid2, feat_mean)]
                diff = flow.pyramid.compose_pyramid(pyramid_diff).mean(1, keepdim=True)# b,1,h,w
                diff_list.append(diff.cpu())
                labels_list.append(labels.cpu()==1)# b,1,h,w

        labels_all = torch.concat(labels_list, dim=0)# b1hw 
        amaps = torch.concat(diff_list, dim=0)# b1hw 
        amaps, labels_all = amaps[:, 0], labels_all[:, 0] # both b,h,w
        pixel_auroc = roc_auc_score(labels_all.flatten(), amaps.flatten()) # pixel score
        image_auroc = roc_auc_score(labels_all.amax((-1,-2)), amaps.amax((-1,-2))) # image score
        pixel_pro = compute_pro_score_fast(amaps, labels_all) # pro score
        logger.info(f'   TEST Pixel-AUROC: {pixel_auroc}, time: {time.time()-t0:.1f}s')
        logger.info(f'   TEST Image-AUROC: {image_auroc}, time: {time.time()-t0:.1f}s')
        logger.info(f'   TEST Pixel-PRO: {pixel_pro}, time: {time.time()-t0:.1f}s')

        if pixel_auroc > np.max(pixel_auroc_lst):
            save_dict['state_dict_pixel'] = {k: v.cpu() for k, v in flow.state_dict().items()} # save ckpt
        if pixel_pro > np.max(pixel_pro_lst):
            save_dict['state_dict_pro'] = {k: v.cpu() for k, v in flow.state_dict().items()} # save ckpt
        pixel_auroc_lst.append(pixel_auroc)
        pixel_pro_lst.append(pixel_pro)
        image_auroc_lst.append(image_auroc)

        del amaps, labels_all, diff_list, labels_list
    save_dict['pixel_auroc_lst'] = pixel_auroc_lst
    save_dict['image_auroc_lst'] = image_auroc_lst
    save_dict['pixel_pro_lst']   = pixel_pro_lst
    save_dict['losses_lst'] = losses_lst
    np.savez(f'saveDir/{save_name}.npz', **save_dict) # save all 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training on MVTecAD')
    parser.add_argument('--cls', type=str, default='tile', choices=\
                        ['tile', 'leather', 'hazelnut', 'toothbrush', 'wood', 'bottle', 'cable', \
                         'capsule', 'pill', 'transistor', 'carpet', 'zipper', 'grid', 'screw', 'metal_nut'])
    parser.add_argument('--datapath', type=str, default='../mvtec_anomaly_detection')
    # hyper-parameters of architecture
    parser.add_argument('--encoder', type=str, default='resnet18', choices=['none', 'resnet18', 'resnet34'])
    parser.add_argument('--numLayer', type=str, default='auto', choices=['auto', '2', '4', '8'])
    parser.add_argument('--volumeNorm', type=str, default='auto', choices=['auto', 'CVN', 'SVN'])
    # non-key parameters of architecture
    parser.add_argument('--kernelSize', type=int, default=7, choices=[3, 5, 7, 9, 11])
    parser.add_argument('--numChannel', type=int, default=16)
    parser.add_argument('--numStack', type=int, default=4)
    # other parameters
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batchSize', type=int, default=2)
    parser.add_argument('--saveMemory', type=bool, default=True) 
    
    args = parser.parse_args()
    cls_name = args.cls
    resnetX = 0 if args.encoder=='none' else int(args.encoder[6:])
    if args.volumeNorm == 'auto':
        vn_dims = (0, 2, 3) if cls_name in ['carpet', 'grid', 'bottle', 'transistor'] else (0, 1)
    elif args.volumeNorm == 'CVN':
        vn_dims = (0, 1)
    elif args.volumeNorm == 'SVN':
        vn_dims = (0, 2, 3)
    if args.numLayer == 'auto':
        num_layer = 4
        if cls_name in ['metal_nut', 'carpet', 'transistor']:
            num_layer = 8
        elif cls_name in ['screw',]:
            num_layer = 2
    else:
        num_layer = int(args.numLayer)
    ksize = args.kernelSize
    numChannel = args.numChannel
    numStack = args.numStack
    gpu_id = args.gpu
    batchSize = args.batchSize
    saveMem = args.saveMemory
    datapath = args.datapath

    logger, save_name = getLogger(f'./saveDir')
    logger.info(f'========== Config ==========')
    logger.info(f'> Class: {cls_name}')
    logger.info(f'> MVTecAD dataset root: {datapath}')
    logger.info(f'> Encoder: {args.encoder}')
    logger.info(f"> Volume Normalization: {'CVN' if len(vn_dims)==2 else 'SVN'}")
    logger.info(f'> Num of Pyramid Layer: {num_layer}')
    logger.info(f'> Conv Kernel Size in NF: {ksize}')
    logger.info(f'> Num of Channels in NF: {numChannel}')
    logger.info(f'> Num of Stack Block: {numStack}')
    logger.info(f'> Batch Size: {batchSize}')
    logger.info(f'> GPU device: cuda:{gpu_id}')
    logger.info(f'> Save Training Memory: {saveMem}')
    logger.info(f'============================')

    train(logger, save_name, cls_name, datapath,\
          resnetX, num_layer, vn_dims, \
          ksize=ksize, channel=numChannel, num_stack=numStack, \
          device=f'cuda:{gpu_id}', batch_size=batchSize, save_memory=saveMem)

