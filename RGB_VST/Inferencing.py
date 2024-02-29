from sympy import false
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import tqdm
from dataset import transform_only
import transforms as trans
from torchvision import transforms
import time
from Models.ImageDepthNet import ImageDepthNet
import numpy as np
import os
import sys
import cv2
from torchvision import  transforms, models
from itertools import compress
from PIL import Image
sys.path.append('/opt/data/private/zsf/VST_railway/Classification/')
from model import classifier, RoI_Head

sys.path.append('/opt/data/private/zsf/VST_railway')
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking




def diff_video(vid_frames):
    """
    Compute the difference of each frame in the video with the first frame.
    
    Args:
    first_frame (np.array): The first frame of the video.
    video (cv2.VideoCapture): The video capture object.

    Returns:
    list: A list of frames showing the difference with the first frame.
    """
    ref = 15
    
    num_frames = len(vid_frames)
    diff_frames = []

    for i in range(num_frames):
        if i < ref:
            # If it's one of the first 30 frames, use the first frame as a reference
            reference_frame = vid_frames[0]
        else:
            # Calculate the average of the preceding 30 frames
            reference_frame = np.mean(vid_frames[i-ref:i], axis=0).astype(np.uint8)

        # Compute the absolute difference
        diff = cv2.absdiff(reference_frame, vid_frames[i])
        diff_frames.append(diff)

    return diff_frames

def get_proposal(mask, img, resize):
    bbox_imgs = []
    bboxes = []
    # print(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤面积小于某个阈值的区域
    area_threshold = 100  # 根据需要调整阈值
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > area_threshold]
    # print("len(large_contours):", len(large_contours))
    # 创建一个新的mask来保存过滤后的区域
    filtered_mask = np.zeros_like(mask)

    for cnt in large_contours:
        # 对每个大区域获取最小外接矩形
        x,y,w,h = cv2.boundingRect(cnt)
        # 在mask上填充矩形
        bbox_img = img[y:y+h, x:x+w, :]
        # print(bbox_img.shape)
        bbox_img = cv2.resize(bbox_img, (resize,resize))
        bbox_imgs.append(bbox_img)
        bboxes.append([x,y,w,h])
    
    return bbox_imgs, bboxes
    

def infer_net(args):

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

    pth_path = '/opt/data/private/zsf/VST_railway/Classification/final_res34_finetune.pth'
    state_dict = torch.load(pth_path)
    classifier_model.load_state_dict(state_dict)
    print('Classifier loaded from {}'.format(pth_path))
    classifier_model.cuda()
    classifier_model.eval()
    
    
    # Inin MOT model
    print('<<<<<<<Tracking model loadeding!>>>>>>>')
    args.track_thresh = 0.5
    args.mot20 = false
    args.track_buffer = 30 # 30
    args.match_thresh = 0.8 # 0.8
    args.fps = 30
    tracker = BYTETracker(args, frame_rate=args.fps)

    
    # Processing the video input
    video = cv2.VideoCapture(args.video_input)
        
    vid_frames = []
    print('<<Loding the video frames!>>')
    frame_num = 0
    while video.isOpened():
        print(f"Reading frame{frame_num}!")
        success, frame = video.read()
        frame_num += 1
        if success:
            vid_frames.append(frame)
        else:
            break

    diff_frames = diff_video(vid_frames)
    video.release()

    
    print('''
                Starting testing:
                    Testing video frame length: {}
                '''.format(len(diff_frames)))

    time_list = []
    visualized_output = []
    results = []
    for i, frame in tqdm.tqdm(enumerate(diff_frames)):
        # frame = np.expand_dims(frame, axis=0)
        # print(frame.shape)
        images, image_w, image_h = transform_only(frame, args.img_size)
        images = Variable(images.cuda())

        starts = time.time()
        images = images.unsqueeze(0)
        # print(images.shape)
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
        bbox_imgs, bboxes = get_proposal(mask_bool, vid_frames[i], 64)
        if len(bbox_imgs)==0:
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
        print("bboxes:", bboxes)
        print('class_preds:', class_preds)
        # indices = (class_preds[:, 1] > 0.6) # filter by cls head
        indices = (class_preds[:, 1] > 0) # all proposal box, note that if using tracking module there should send all box to it
        # print('indices:', indices)
        probs = class_preds[indices]
        selected_bbox = list(compress(proposals['bbox'], indices.tolist()))
        
        detections = []
        for bbox, prob in zip(selected_bbox, probs):
            # cv2.rectangle(vid_frames[i], (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255,0,0), thickness=2)
            # cv2.putText(vid_frames[i], f'prob:{prob[0]}', (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            detection = bbox + [prob[1].item()]  
            detections.append(detection)
        
        if detections is not None:
            online_targets = tracker.update(torch.tensor(detections), [image_h, image_w], (image_h, image_w))
            online_tlwhs = []
            online_ids = []
            online_scores = []
            # for t in online_targets:
            #     tlwh = t.tlwh
            #     tid = t.track_id
            #     # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            #     # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
            #     online_tlwhs.append(tlwh)
            #     online_ids.append(tid)
            #     online_scores.append(t.score)
            #     # save results
            #     results.append(
            #         f"{i},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
            #     )
            for t in tracker.removed_stracks:
                if t.end_frame - t.start_frame >= 5:
                    tlwh = t.tlwh
                    tid = t.track_id
                    # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{i},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            ends = time.time()
            time_use = ends - starts
            time_list.append(time_use)
            online_im = plot_tracking(
                vid_frames[i], online_tlwhs, online_ids, frame_id=i, fps=1. / time_use
            )
        else:
            online_im = vid_frames[i]
        visualized_output.append(online_im) # result after tracking


        # visualized_output.append(vid_frames[i]) # output
        # visualized_output.append(diff_frames[i]) # diff_img
        # visualized_output.append(mask_binary_3ch) # salient detection mask
        # cv2.imwrite(f'diff_frame_{i}.jpg', diff_frames[i])

    

    # save saliency maps
    save_test_path = args.save_test_path_root + '/video/'
    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path)
        
        
    cap = cv2.VideoCapture(-1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(os.path.join(save_test_path, f"{args.video_output_name}.mp4"), fourcc, 10.0, (image_w, image_h), True)
    for _vis_output in visualized_output:
        frame = _vis_output
        out.write(frame)
    cap.release()
    out.release()

    print('video:{}, cost:{}'.format(args.video_input, np.mean(time_list) * 1000))





