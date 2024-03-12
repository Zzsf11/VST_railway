from torch.utils import data
import os
from PIL import Image


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root):
        pred_names = os.listdir(pred_root)
        #label_names = os.listdir(label_root)

        if pred_root.split('/')[-2] == 'PASCAL-S':
            # remove the following image
            if '424.png' in pred_names:
                pred_names.remove('424.png')
            if '460.png' in pred_names:
                pred_names.remove('460.png')
            if '359.png' in pred_names:
                pred_names.remove('359.png')
            if '408.png' in pred_names:
                pred_names.remove('408.png')
            if '622.png' in pred_names:
                pred_names.remove('622.png')

        # self.image_path = list(
        #     map(lambda x: os.path.join(pred_root, x), pred_names))
        # self.label_path = list(
        #     map(lambda x: os.path.join(label_root, x), pred_names))
        
        # 定义要遍历的目录
        root_dir = '/opt/data/private/zsf/VST_railway/RGB_VST/Pred/test'

        # 初始化一个空列表，用于存放找到的图片路径
        self.image_path = []

        # 定义支持的图片文件扩展名
        image_extensions = {'.jpg', '.jpeg', '.png'}

        # 遍历root_dir下的所有目录和文件
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                # 检查文件扩展名是否是支持的图片格式
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    # 构造文件的完整路径并添加到列表中
                    self.image_path.append(os.path.join(dirpath, filename))
                    
                            # 定义要遍历的目录
        root_dir = '/opt/data/private/zsf/VST_railway/RGB_VST/GT'

        # 初始化一个空列表，用于存放找到的图片路径
        self.label_path = []

        # 遍历root_dir下的所有目录和文件
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                # 检查文件扩展名是否是支持的图片格式
                if os.path.splitext(filename)[1].lower() in image_extensions:
                    # 构造文件的完整路径并添加到列表中
                    self.label_path.append(os.path.join(dirpath, filename))


    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
