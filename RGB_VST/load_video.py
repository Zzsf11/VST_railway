import cv2
import sys
import os
import numpy as np

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


video = cv2.VideoCapture("/opt/data/private/zsf/VST_railway/RGB_VST/扔箱子_白天_光线充足_雨_50m_无干扰20230706_125728.mp4")
    
vid_frames = []
print('<<Loding the video frames!>>')
while video.isOpened():
    success, frame = video.read()
    if success:
        vid_frames.append(frame)
    else:
        break

diff_frames = diff_video(vid_frames)
video.release()