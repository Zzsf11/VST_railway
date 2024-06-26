import os
import torch
import Training
import Testing
import Inferencing
import Get_pred
import Get_pred_img
import Get_pred_img_box
import Get_pred_img_box_track
import Get_pred_img_box_track_tracking
import Get_pred_img_box_track_tracking_onlylast
import Infer_on_img
from Evaluation import main
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
    parser.add_argument('--data_root', default='./Data/', type=str, help='data path')
    parser.add_argument('--train_steps', default=60000, type=int, help='total training steps') # 60000
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    parser.add_argument('--pretrained_model', default='./pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str, help='load Pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    # parser.add_argument('--lr', default=2e-4, type=int, help='learning rate') # 1e-4
    # parser.add_argument('--lr', default=2e-5, type=int, help='learning rate') # 1e-4
    parser.add_argument('--lr', default=1e-5, type=int, help='learning rate') # 1e-4
    parser.add_argument('--epochs', default=200, type=int, help='epochs')
    parser.add_argument('--batch_size', default=12, type=int, help='batch_size') # 12
    parser.add_argument('--stepvalue1', default=30000, type=int, help='the step 1 for adjusting lr') # 30000
    parser.add_argument('--stepvalue2', default=45000, type=int, help='the step 2 for adjusting lr') # 45000
    parser.add_argument('--trainset', default='DUTS/DUTS-TR', type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default='checkpoint/', type=str, help='save model path')
    
    
    parser.add_argument("--team_name", default="zzsf123", type=str)
    parser.add_argument("--project_name", default="SOD", type=str)
    parser.add_argument("--experiment_name", default="diff_single_3000", type=str)

    # test
    parser.add_argument('--Testing', default=False, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default='preds/', type=str, help='save saliency maps path')
    # parser.add_argument('--test_paths', type=str, default='DUTS/DUTS-TE+ECSSD+HKU-IS+PASCAL-S+DUT-O+BSD')
    parser.add_argument('--test_paths', type=str, default='DUTS/DUTS-TE')

    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='RGB_VST', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default='./', help='path for saving result.txt')


    parser.add_argument('--Inferencing', default=False, type=bool, help='Inferencing or not')
    parser.add_argument('--video_input', type=str, default='video')
    parser.add_argument('--video_output_name', type=str, default='1')
    
    parser.add_argument('--classifier_pth_path', type=str, default='/opt/data/private/zsf/VST_railway/Classification/checkpoints/with_pretrain/final_res34_finetune.pth')
    
    # parser.add_argument('--diff_frame_folder', type=str, default='/opt/data/private/zsf/VST_railway/RGB_VST/Data/diff_random1')
    # parser.add_argument('--diff_frame_folder', type=str  , default='/opt/data/private/zsf/VST_railway/RGB_VST/Data/diff_video_ref15')
    # parser.add_argument('--diff_frame_folder', type=str, default='/opt/data/private/zsf/VST_railway/RGB_VST/Data/diff_video_ref15_0-14')
    # parser.add_argument('--diff_frame_folder', type=str, default='/opt/data/private/zsf/VST_railway/RGB_VST/Data/diff_video_refavg')
    parser.add_argument('--diff_frame_folder', type=str, default='/opt/data/private/zsf/VST_railway/RGB_VST/Data/diff_video_ref_0')
    
    parser.add_argument('--Get_pred', default=False, type=bool, help='Inferencing or not')
    parser.add_argument('--Get_pred_img', default=False, type=bool, help='Inferencing or not')    
    parser.add_argument('--Get_pred_img_box', default=False, type=bool, help='Inferencing or not')
    parser.add_argument('--Get_pred_img_box_track', default=False, type=bool, help='Inferencing or not')
    parser.add_argument('--Get_pred_img_box_track_tracking', default=False, type=bool, help='Inferencing or not')
    parser.add_argument('--Get_pred_img_box_track_tracking_onlylast', default=False, type=bool, help='Inferencing or not')
    parser.add_argument('--data_path', type=str, default='/opt/data/private/zsf/Railway/part4/allscenes')

    
    parser.add_argument('--Infer_on_img', default=False, type=bool, help='Inferencing or not')
    parser.add_argument('--Infer_on_img_out', type=str, default='/opt/data/private/zsf/VST_railway/RGB_VST/Infer_on_img')
    
    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')
    
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    num_gpus = torch.cuda.device_count()
    print(f"num_gpus{num_gpus}")
    if args.Training:
        Training.train_net(num_gpus=num_gpus, args=args)
    if args.Testing:
        Testing.test_net(args)
    if args.Evaluation:
        main.evaluate(args)
    if args.Inferencing:
        Inferencing.infer_net(args)
    if args.Get_pred:
        Get_pred.get_pred(args)
    if args.Get_pred_img:
        Get_pred_img.get_pred(args)
    if args.Get_pred_img_box:
        Get_pred_img_box.get_pred(args)
    if args.Get_pred_img_box_track:
        Get_pred_img_box_track.get_pred(args)
    if args.Get_pred_img_box_track_tracking:
        Get_pred_img_box_track_tracking.get_pred(args)
    if args.Get_pred_img_box_track_tracking_onlylast:
        Get_pred_img_box_track_tracking_onlylast.get_pred(args)
    if args.Infer_on_img:
        Infer_on_img.get_pred(num_gpus=num_gpus, args=args)