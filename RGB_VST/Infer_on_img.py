import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import transforms as trans
from torchvision import transforms
from dataset import get_loader
import torch.distributed as dist
from Models.ImageDepthNet import ImageDepthNet
import torch.multiprocessing as mp
import os
import cv2
import numpy as np


def get_pred(num_gpus, args):

    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))
    
def main(local_rank, num_gpus, args):    
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method=args.init_method, world_size=num_gpus, rank=local_rank)

    torch.cuda.set_device(local_rank)

    net = ImageDepthNet(args)
    net.cuda()
    net.eval()

    # Init Salient detection model
    # load model (multi-gpu)
    model_path = args.save_model_dir
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    print('Salient detection model loaded from {}'.format(model_path))
    
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True)
    
    test_dataset = get_loader(args.test_paths, args.data_root, args.img_size, mode='test')
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=num_gpus,
        rank=local_rank,
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=6,
                                               pin_memory=True,
                                               sampler=sampler,
                                               drop_last=True,
                                               )

    print('''
        Starting testing:
            Batch size: {}
            Training size: {}
        '''.format(args.batch_size, len(test_loader.dataset)))
    
    
    if not os.path.exists(args.Infer_on_img_out):
        os.makedirs(args.Infer_on_img_out)
        
    
    for i, data_batch in enumerate(test_loader):
        
        images, w, h, img_path = data_batch
        images = Variable(images.cuda())

        outputs_saliency, _ = net(images)

        _, _, _, mask_1_1 = outputs_saliency


        output_s = F.sigmoid(mask_1_1)
        
        

        for i in range(output_s.size(0)):
            # 单独获取每个图像
            out = output_s[i].unsqueeze(0)  # 增加一个批次维度，以匹配 interpolate 函数的输入要求
            target_height = h[i].item()
            target_width = w[i].item()

            # 插值
            interpolated_out = F.interpolate(out, size=(target_height, target_width), mode='bilinear', align_corners=False)
            # 去掉批次维度并转换为NumPy数组
            out_np = interpolated_out.squeeze().detach().cpu().numpy()
            
            # 数据类型转换，确保数据类型为uint8
            out_np_uint8 = (out_np * 255).astype('uint8')
            # _, mask_bool = cv2.threshold(out_np_uint8, 0, 255, cv2.THRESH_BINARY)
            
            # 将插值后的图像添加到列表中
            # interpolated_images.append(interpolated_image)
            out_name = os.path.basename(img_path[i])
            save_path = os.path.join(args.Infer_on_img_out, out_name)
            cv2.imwrite(save_path, out_np_uint8)