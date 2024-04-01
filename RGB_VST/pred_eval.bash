# python train_test_eval.py --Get_pred True --save_model_dir checkpoint/harmonization_cv2diff8000RGB_VST.pth 
# python train_test_eval.py  --Evaluation True  --data_root /opt/data/private/zsf/VST_railway/RGB_VST/GT/ --save_test_path_root /opt/data/private/zsf/VST_railway/RGB_VST/Pred/test/ --test_paths harmonization_cv2diff8000RGB


# python train_test_eval.py --Get_pred True --save_model_dir checkpoint/Mydataset_harmonization_cv2diff_avg6000RGB_VST.pth
# python train_test_eval.py  --Evaluation True  --data_root /opt/data/private/zsf/VST_railway/RGB_VST/GT/ --save_test_path_root /opt/data/private/zsf/VST_railway/RGB_VST/Pred/test/ --test_paths Mydataset_harmonization_cv2diff6000RGBVST_Get_pred

python train_test_eval.py --Get_pred_img True --save_model_dir checkpoint/Mydataset_harmonization_cv2diff_avg6000RGB_VST.pth
python train_test_eval.py  --Evaluation True  --data_root /opt/data/private/zsf/VST_railway/RGB_VST/GT/ --save_test_path_root /opt/data/private/zsf/VST_railway/RGB_VST/Pred/test/ --test_paths Mydataset_harmonization_cv2diff6000RGBVST_Get_pred_img

python train_test_eval.py --Get_pred_img_box True --save_model_dir checkpoint/Mydataset_harmonization_cv2diff_avg6000RGB_VST.pth
python train_test_eval.py  --Evaluation True  --data_root /opt/data/private/zsf/VST_railway/RGB_VST/GT/ --save_test_path_root /opt/data/private/zsf/VST_railway/RGB_VST/Pred/test/ --test_paths Mydataset_harmonization_cv2diff6000RGBVST_Get_pred_img_box

python train_test_eval.py --Get_pred_img_box_track True --save_model_dir checkpoint/Mydataset_harmonization_cv2diff_avg6000RGB_VST.pth
python train_test_eval.py  --Evaluation True  --data_root /opt/data/private/zsf/VST_railway/RGB_VST/GT/ --save_test_path_root /opt/data/private/zsf/VST_railway/RGB_VST/Pred/test/ --test_paths Mydataset_harmonization_cv2diff6000RGBVST_Get_pred_img_box_track

python train_test_eval.py --Get_pred_img_box_track_tracking True --save_model_dir checkpoint/Mydataset_harmonization_cv2diff_avg6000RGB_VST.pth
python train_test_eval.py  --Evaluation True  --data_root /opt/data/private/zsf/VST_railway/RGB_VST/GT/ --save_test_path_root /opt/data/private/zsf/VST_railway/RGB_VST/Pred/test/ --test_paths Mydataset_harmonization_cv2diff6000RGBVST_Get_pred_img_box_track_tracking

python train_test_eval.py --Get_pred_img_box_track_tracking_onlylast True --save_model_dir checkpoint/Mydataset_harmonization_cv2diff_avg6000RGB_VST.pth
python train_test_eval.py  --Evaluation True  --data_root /opt/data/private/zsf/VST_railway/RGB_VST/GT/ --save_test_path_root /opt/data/private/zsf/VST_railway/RGB_VST/Pred/test/ --test_paths Mydataset_harmonization_cv2diff6000RGBVST_Get_pred_img_box_track_tracking_onlylast
