import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm
from dataset import transform_only
import transforms as trans
from torchvision import transforms
from Models.ImageDepthNet import ImageDepthNet
from torchvision import  transforms, models
from itertools import compress
import numpy as np
import os
import cv2
from pycocotools.coco import COCO
import random
from PIL import Image
from model import classifier, RoI_Head


'''
SOD + classify = box

得到视频抽帧样本的pred和GT
不使用视频抽帧，通过随机选取序列中的一帧做ref

'''
def is_bbox_inside_mask(bbox, mask):
    bbox = np.array(bbox).astype(int)
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    
    # 计算框的中心点坐标
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # 检查中心点是否在掩码区域内
    is_inside = mask[center_y, center_x]
    
    return is_inside

def get_masked_bboxes(bboxes, ids, masks):
    masked_bboxes = []
    masked_ids = []
    for bbox, id in zip(bboxes, ids):
            
        if is_bbox_inside_mask(bbox, masks) == False:
            continue
        masked_bboxes.append(bbox)
        masked_ids.append(id)
    
    return masked_bboxes, masked_ids

def diff_video(vid_frames):
    """
    计算视频中每一帧与随机一帧的差异。

    参数:
    vid_frames (list of np.array): 视频帧的列表。
    
    返回:
    list: 与显示差异的帧列表。
    """    
    diff_frames = []

    for i in range(len(vid_frames)):
        # 从vid_frames中随机选择一帧，但不包括当前帧i
        other_frame_index = random.choice([x for x in range(len(vid_frames)) if x != i])
        
        # 计算当前帧和随机选中的帧之间的差分
        diff = cv2.absdiff(vid_frames[i], vid_frames[other_frame_index])
        
        # 将差分保存到diff_frames列表中
        diff_frames.append(diff)

    return diff_frames


def get_proposal(mask):
    bboxes = []
    # print(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤面积小于某个阈值的区域
    area_threshold = 100  # 根据需要调整阈值
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]

    for cnt in large_contours:
        # 对每个大区域获取最小外接矩形
        x,y,w,h = cv2.boundingRect(cnt)

        bboxes.append([x,y,w,h])
    
    return bboxes

def boxes2mask(boxes, img_size):
    # 创建空白图像
    img_width, img_height = img_size[:]
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # 定义mask区域颜色(BGR格式)
    mask_color = (255, 255, 255)  # 白色

    # 遍历bbox列表,画出mask区域
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), mask_color, -1)
        
    return img

def get_pred(args):

    cudnn.benchmark = True

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
    
     # Init Classifier
    backbone = models.resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 512
    roi_head = RoI_Head(None, None, None, out_channels=backbone.out_channels, num_classes=2)
    classifier_model = classifier(backbone, roi_head, mode='test')
    pth_path = args.classifier_pth_path
    state_dict = torch.load(pth_path)
    classifier_model.load_state_dict(state_dict)
    print('Classifier loaded from {}'.format(pth_path))
    classifier_model.cuda()
    classifier_model.eval()
    
    
    for video in os.listdir('/opt/data/private/zsf/Railway/part4/allscenes'):
        file_name_with_extension = os.path.basename(video)
        
        # 使用str.rsplit分割字符串，最多分割一次，然后选择第一个元素得到文件名
        ann_path = os.path.join('/opt/data/private/zsf/Railway/part4/allscenes', video,  'trainnew.json')
        ann = COCO(ann_path)
        img_ids = ann.getImgIds()
        
        
        mask_folder = os.path.join('/opt/data/private/zsf/VST_railway/RGB_VST/GT', file_name_with_extension)
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)
        
        # get frame_index
        img_name = []
        
        imgs= []
        for img_id in img_ids:
            img_info = ann.loadImgs(img_id)[0]
        
            # save gt mask
            mask = np.zeros([img_info['height'], img_info['width']])
            ann_id = ann.getAnnIds(imgIds=img_id, catIds=2)
            ann_info = ann.loadAnns(ann_id)
            for i, _ in enumerate(ann_id):
                if ann_info[i]['segmentation'] != []:
                    mask += ann.annToMask(ann_info[i])
            mask_name = os.path.join(mask_folder, img_info['file_name'])    
            cv2.imwrite(mask_name, mask*255)
            img_name.append(img_info['file_name'])
            
            img_path = os.path.join('/opt/data/private/zsf/Railway/part4/allscenes', video, 'train/', img_info['file_name'])
            imgs.append(cv2.imread(img_path))
            
        # 保存diff_frames
        # diff_frames = diff_video(imgs)
        # diff_folder = os.path.join('/opt/data/private/zsf/VST_railway/RGB_VST/Data/diff_random1', file_name_with_extension)
        # if not os.path.exists(diff_folder):
        #     os.makedirs(diff_folder)
        
        # 使用保存好的diff_frames
        diff_frames = []
        diff_frame_folder = os.path.join(args.diff_frame_folder, file_name_with_extension)
        for diff_frame in os.listdir(diff_frame_folder):
            diff_path = os.path.join(diff_frame_folder, diff_frame)
            diff_frames.append(cv2.imread(diff_path))
        
        print('''
                    Starting testing:
                        Testing video frame length: {}
                    '''.format(len(diff_frames)))

        time_list = []
        
        pred_folder = os.path.join('/opt/data/private/zsf/VST_railway/RGB_VST/Pred/test', file_name_with_extension)
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
            
            
        
        for i, frame in tqdm.tqdm(enumerate(diff_frames)):

            images, image_w, image_h = transform_only(frame, args.img_size)
            images = Variable(images.cuda())

            images = images.unsqueeze(0)

            outputs_saliency, _ = net(images)

            _, _, _, mask_1_1 = outputs_saliency

            image_w, image_h = int(image_w), int(image_h)

            output_s = F.sigmoid(mask_1_1)

            output_s = output_s.data.cpu().squeeze(0)

            transform = trans.Compose([
                transforms.ToPILImage(),
                trans.Scale((image_w, image_h))
            ])
            output_s = transform(output_s)
            output_s = np.array(output_s)
            _, mask_bool = cv2.threshold(output_s, 0, 255, cv2.THRESH_BINARY)
            # mask_binary_3ch = np.dstack([mask_bool, mask_bool, mask_bool]).astype(np.uint8) # for vis mask
            bboxes = get_proposal(mask_bool)
            if len(bboxes)==0:
                continue
            
            # Normalization
            transform = transforms.Compose([
                    # transforms.Resize([800, 1333]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            frame_pil = Image.fromarray(frame.astype('uint8'), 'RGB')
            img = transform(frame_pil).cuda().unsqueeze(0)
            proposals = {}
            proposals['bbox'] = [[x_min, y_min, x_min + width, y_min + height] for x_min, y_min, width, height in bboxes]
            class_logits = classifier_model(img, proposals['bbox'])
            class_preds = F.softmax(class_logits, dim=1)
            # print("bboxes:", bboxes)
            # print('class_preds:', class_preds)
            indices = (class_preds[:, 1] > 0.6) # filter by cls head
            # indices = (class_preds[:, 1] > 0) # all proposal box, note that if using tracking module there should send all box to it
            # print('indices:', indices)
            probs = class_preds[indices]
            selected_bbox = list(compress(proposals['bbox'], indices.tolist()))
            
            pred = boxes2mask(selected_bbox, [image_w, image_h])
            
            pred_name = os.path.join(pred_folder, img_name[i])    
            cv2.imwrite(pred_name, pred)
            
            # diff_name = os.path.join(diff_folder, img_name[i])  
            # cv2.imwrite(diff_name, frame)
                        
            # gray_img_rgb = cv2.cvtColor(mask_bool, cv2.COLOR_GRAY2RGB)
            # combined_img = np.hstack((frame, gray_img_rgb))
            # cv2.imwrite(pred_name, combined_img)
            
            # masked
            # alpha = 0.5  # 设定透明度
            # result_img = imgs[i].copy()

            # # 将mask_bool转换为三通道，以便与result_img进行叠加
            # # mask_rgb = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)

            # # 将mask_rgb应用于result_img以实现透明效果
            # combined_img = cv2.addWeighted(pred, alpha, result_img, 1 - alpha, 0, result_img)
                        
            # cv2.imwrite(pred_name, combined_img)
            
            

        print('video:{}, cost:{}'.format(args.video_input, np.mean(time_list) * 1000))





