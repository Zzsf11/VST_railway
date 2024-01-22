import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm
from dataset import get_loader, transform_only
import transforms as trans
from torchvision import transforms
import time
from Models.ImageDepthNet import ImageDepthNet
from torch.utils import data
import numpy as np
import os
import cv2

def diff_video(vid_frames):
    """
    Compute the difference of each frame in the video with the first frame.
    
    Args:
    first_frame (np.array): The first frame of the video.
    video (cv2.VideoCapture): The video capture object.

    Returns:
    list: A list of frames showing the difference with the first frame.
    """
    first_frame = vid_frames[0]
    diff_frames = []

    for frame in vid_frames:
        diff = cv2.absdiff(first_frame, frame)
        diff_frames.append(diff)
    
    return diff_frames

    

def infer_net(args):

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
    print('Model loaded from {}'.format(model_path))

    # load model
    # net.load_state_dict(torch.load(model_path))
    # model_dict = net.state_dict()
    # print('Model loaded from {}'.format(model_path))
    
    # Processing the video input
    video = cv2.VideoCapture(args.video_input)
        
    vid_frames = []
    while video.isOpened():
        success, frame = video.read()
        if success:
            vid_frames.append(frame)
        else:
            break

    diff_frames = diff_video(vid_frames)
    video.release()
    # diff_frames = vid_frames
    
    print('''
                Starting testing:
                    Testing video frame length: {}
                '''.format(len(diff_frames)))

    time_list = []
    visualized_output = []
    for i, frame in tqdm.tqdm(enumerate(diff_frames)):
        # frame = np.expand_dims(frame, axis=0)
        # print(frame.shape)
        images, image_w, image_h = transform_only(frame, args.img_size)
        images = Variable(images.cuda())

        starts = time.time()
        images = images.unsqueeze(0)
        # print(images.shape)
        outputs_saliency, outputs_contour = net(images)
        ends = time.time()
        time_use = ends - starts
        time_list.append(time_use)

        mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency

        image_w, image_h = int(image_w), int(image_h)

        output_s = F.sigmoid(mask_1_1)

        output_s = output_s.data.cpu().squeeze(0)

        transform = trans.Compose([
            transforms.ToPILImage(),
            trans.Scale((image_w, image_h))
        ])
        output_s = transform(output_s)
        output_s = np.array(output_s)
        mask_bool = output_s.astype(bool)
        mask_binary_3ch = np.dstack([mask_bool, mask_bool, mask_bool]).astype(np.uint8)
        visualized_output.append(mask_binary_3ch * vid_frames[i])

    

    # save saliency maps
    save_test_path = args.save_test_path_root + '/video/'
    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path)
        
        
    cap = cv2.VideoCapture(-1)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(os.path.join(save_test_path, "visualization.mp4"), fourcc, 10.0, (image_w, image_h), True)
    for _vis_output in visualized_output:
        frame = _vis_output
        out.write(frame)
    cap.release()
    out.release()

    print('video:{}, cost:{}'.format(args.video_input, np.mean(time_list) * 1000))





