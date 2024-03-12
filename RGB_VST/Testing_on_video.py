import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import transform_only
import transforms as trans
from torchvision import transforms
import time
from Models.ImageDepthNet import ImageDepthNet
from torch.utils import data
import numpy as np
import os
import cv2
import tqdm
from PIL import Image


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
        # if i < ref:
        #     # If it's one of the first 30 frames, use the first frame as a reference
        #     reference_frame = vid_frames[0]
        # else:
        #     # Calculate the average of the preceding 30 frames
        #     reference_frame = np.mean(vid_frames[i-ref:i], axis=0).astype(np.uint8)
        reference_frame = vid_frames[0]
        # Compute the absolute difference
        diff = cv2.absdiff(reference_frame, vid_frames[i])
        diff_frames.append(diff)


    return diff_frames

def test_net(args):

    cudnn.benchmark = True

    net = ImageDepthNet(args)
    net.cuda()
    net.eval()

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
    # load model
    # net.load_state_dict(torch.load(model_path))
    # model_dict = net.state_dict()
    # print('Model loaded from {}'.format(model_path))

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

        
           
            
        


            

    print('video:{}, cost:{}'.format(args.video_input, np.mean(time_list) * 1000))


