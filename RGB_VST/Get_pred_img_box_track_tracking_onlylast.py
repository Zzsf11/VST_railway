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
import json
import sys
from nltk.metrics.distance import edit_distance
from pycocotools.coco import COCO
import random
from PIL import Image
from model import classifier, RoI_Head
sys.path.append('/opt/data/private/zsf/VST_railway')
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking

'''
SOD + classify + track region fileter = 铁轨区域内的box
再加入tracking 
只保存最后一帧

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

def get_masked_bboxes(bboxes, masks):
    masked_bboxes = []
    for bbox in bboxes:
            
        if is_bbox_inside_mask(bbox, masks) == False:
            continue
        masked_bboxes.append(bbox)
    
    return masked_bboxes

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
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), mask_color, -1)
        
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
    
    # Inin MOT model
    print('<<<<<<<Tracking model loadeding!>>>>>>>')
    args.track_thresh = 0.6
    args.mot20 = False
    args.track_buffer = 3 # 30
    args.match_thresh = 0.8 # 0.8
    args.fps = 1
    tracker = BYTETracker(args, frame_rate=args.fps)
    
    
    
    
    for video in os.listdir('/opt/data/private/zsf/Railway/part4/allscenes'):
        file_name_with_extension = os.path.basename(video)
        
        # 使用str.rsplit分割字符串，最多分割一次，然后选择第一个元素得到文件名
        ann_path = os.path.join('/opt/data/private/zsf/Railway/part4/allscenes', video,  'trainnew.json')
        ann = COCO(ann_path)
        img_ids = ann.getImgIds()
        
        print('Finding the mask for video!')
        with open('/opt/data/private/zsf/Railway/part4/test_mask/_annotations.coco.json') as f:
            coco_data = json.load(f)

        min_distance = float('inf')
        closest_filename = None
        closest_id = None
        for image in coco_data['images']:
            current_filename = image['file_name'].split(".")[0]
            distance = edit_distance(os.path.basename(file_name_with_extension), current_filename)
            # print(f"current_filename:{current_filename}, distance:{distance}")
            if distance < min_distance:
                min_distance = distance
                closest_filename = image['file_name']
                closest_id = image['id']

        # Retrieve mask annotations for the closest image
        mask_annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] == closest_id]
        # 解码分割数据并创建二进制掩模
        # print(mask_annotations)

        print(f'Mask file:{closest_filename}')
        
        
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
            if i == 0:
                image_size = (image_h, image_w)
                mask = np.zeros(image_size, dtype=np.uint8)
                for ann in mask_annotations:
                    segmentation = ann['segmentation'][0]  # 获取分割数据
                    segmentation = np.array(segmentation).reshape((-1, 2)).astype(np.int32)  # 转换为numpy数组

                    # 在mask上填充区域
                    cv2.fillPoly(mask, [segmentation], 255)
                # mask_4_vis = np.dstack([mask, mask, mask]).astype(np.uint8)


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
            # indices = (class_preds[:, 1] > 0.6) # filter by cls head
            indices = (class_preds[:, 1] > 0) # all proposal box, note that if using tracking module there should send all box to it
            # print('indices:', indices)
            probs = class_preds[indices]
            selected_bbox = list(compress(proposals['bbox'], indices.tolist()))
                        
            detections = []
            for bbox, prob in zip(selected_bbox, probs):
                # cv2.rectangle(vid_frames[i], (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255,0,0), thickness=2)
                # cv2.putText(vid_frames[i], f'prob:{prob[1]}', (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                detection = bbox + [prob[1].item()]  
                detections.append(detection)
            
            if detections is not None:
                online_targets = tracker.update(torch.tensor(detections), [image_h, image_w], (image_h, image_w))
                online_tlwhs = []
                online_ids = []
                online_scores = []
                
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                
                for t in tracker.removed_stracks:
                    #TODO: 这个参数很重要
                    if t.end_frame - t.start_frame >= 5:
                        tlwh = t.tlwh
                        tid = t.track_id
                        # 检查是否已经存在相同的bbox和id
                        if not any(np.all(tlwh == item) for item in online_tlwhs) and tid not in online_ids:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)

                online_tlwhs = get_masked_bboxes(online_tlwhs, mask) 
                # print(online_tlwhs)
                # print(online_ids)
                # online_im = plot_tracking(
                #     vid_frames[i], online_tlwhs, online_ids, frame_id=i, fps=1. / time_use
                # )
                # pred = boxes2mask(online_tlwhs, [image_w, image_h])
                if i == len(diff_frames) - 1:
                    pred = boxes2mask(online_tlwhs, [image_w, image_h])
                    pred_name = os.path.join(pred_folder, img_name[i])    
                    cv2.imwrite(pred_name, pred)
                # else:
            #     # online_im = vid_frames[i]
            #     pred_name = os.path.join(pred_folder, img_name[i])    
            #     cv2.imwrite(pred_name, pred)
     
            
            

        print('video:{}, cost:{}'.format(args.video_input, np.mean(time_list) * 1000))





