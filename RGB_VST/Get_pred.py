import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm
from dataset import transform_only
import transforms as trans
from torchvision import transforms
import time
from Models.ImageDepthNet import ImageDepthNet
import numpy as np
import os
import cv2
from pycocotools.coco import COCO




def diff_video(vid_frames, index):
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

    for i in index:
        if i < ref:
            # If it's one of the first 30 frames, use the first frame as a reference
            reference_frame = vid_frames[0]
        else:
            # Calculate the average of the preceding 30 frames
            reference_frame = np.mean(vid_frames[i-ref:i], axis=0).astype(np.uint8)
        # reference_frame = vid_frames[0]
        # Compute the absolute difference
        diff = cv2.absdiff(reference_frame, vid_frames[i])
        diff_frames.append(diff)


    return diff_frames

    

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
    
    for video in os.listdir('/opt/data/private/zsf/Railway/part4/video'):
        file_name_with_extension = os.path.basename(video)

        # 使用str.rsplit分割字符串，最多分割一次，然后选择第一个元素得到文件名
        video_name = file_name_with_extension.rsplit('.', 1)[0]
        ann_path = os.path.join("/opt/data/private/zsf/Railway/part4/allscenes", video_name, 'trainnew.json')
        ann = COCO(ann_path)
        img_ids = ann.getImgIds()
        
        
        mask_folder = os.path.join('/opt/data/private/zsf/VST_railway/RGB_VST/GT', video_name)
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)
        
        # get frame_index
        frame_index = []
        img_name = []
        for img_id in img_ids:
            img_info = ann.loadImgs(img_id)[0]
            frame_id = img_info['frame_index']
            frame_index.append(frame_id)
        
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

        # Processing the video input
        video_path = os.path.join('/opt/data/private/zsf/Railway/part4/video', video)
        video = cv2.VideoCapture(video_path)
            
        vid_frames = []
        print(f'<<Loding the {video} video frames!>>')
        frame_num = 0
        while video.isOpened():
            # print(f"Reading frame{frame_num}!")
            success, frame = video.read()
            frame_num += 1
            if success:
                vid_frames.append(frame)
            else:
                break

        diff_frames = diff_video(vid_frames, frame_index)
        video.release()

        
        print('''
                    Starting testing:
                        Testing video frame length: {}
                    '''.format(len(diff_frames)))

        time_list = []
        
        pred_folder = os.path.join('/opt/data/private/zsf/VST_railway/RGB_VST/Pred/test', video_name)
        if not os.path.exists(pred_folder):
            os.makedirs(pred_folder)
        
        for i, frame in tqdm.tqdm(enumerate(diff_frames)):
            # frame = np.expand_dims(frame, axis=0)
            # print(frame.shape)
            images, image_w, image_h = transform_only(frame, args.img_size)
            images = Variable(images.cuda())

            starts = time.time()
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
            
            pred_name = os.path.join(pred_folder, img_name[i])    
            cv2.imwrite(pred_name, mask_bool)

        print('video:{}, cost:{}'.format(args.video_input, np.mean(time_list) * 1000))





