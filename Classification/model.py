from torchvision import datasets, transforms, models, _utils
from torchvision.ops import MultiScaleRoIAlign, boxes 
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import cv2
import os
import torch
from torch import nn, Tensor

class classifier(nn.Module):
    def __init__(
            self,
            backbone,
            roi_head,
            fg_iou_thresh: float = 0.8,
            bg_iou_thresh: float = 0.1,
            mode: str = 'train',
        ) -> None:
        super().__init__()
        self.mode = mode
        self.backbone = backbone
        self.roi_head = roi_head

        self.anchor_generator = models.detection.anchor_utils.AnchorGenerator(
            sizes=tuple([(8, 12, 16) for _ in range(5)]),
            aspect_ratios=tuple([(0.5, 1.0, 1.5) for _ in range(5)]))
        
        # self.anchor_generator = models.detection.anchor_utils.AnchorGenerator(
        #     sizes=((4, 8, 16),),
        #     aspect_ratios=((0.5, 1.0, 2.0),),
        # )

        
        self.box_similarity = boxes.box_iou
        self.proposal_matcher = models.detection._utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )
                
    def visualize_positive_anchors_cv2(self, image, anchors, save_path):
        """使用cv2可视化并将图像上的正样本锚点框保存到文件"""
        # 如果图像有标准化或其他预处理，请在此处进行相应的逆变换
        image = image.permute(1, 2, 0).cpu().numpy().astype('uint8')
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 如果图像是RGB，转换为BGR

        for anchor in anchors:
            x1, y1, x2, y2 = anchor.tolist()
            # 在图像上绘制矩形框
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 保存图像
        cv2.imwrite(save_path, image)
    def vis(self, x, anchors, sampled_positive_anchors, matched_gt_boxes, sampled_negative_anchors):
        vis_folder = os.path.join('/opt/data/private/zsf/VST_railway/Classification','vis_anchor')
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)
        for num in range(x.shape[0]):
            self.visualize_positive_anchors_cv2(x[num], anchors[num],vis_folder+'/'+f"anchor{num}.jpg")
            self.visualize_positive_anchors_cv2(x[num], sampled_positive_anchors[num],vis_folder+'/'+f"pos{num}.jpg")
            self.visualize_positive_anchors_cv2(x[num], matched_gt_boxes[num],vis_folder+'/'+f"gt{num}.jpg")
            self.visualize_positive_anchors_cv2(x[num], sampled_negative_anchors[num],vis_folder+'/'+f"neg{num}.jpg")
        
    def forward(self, x, targets):
        
        if self.mode == 'train':
            feature = self.backbone(x)
            img_shape = [a.shape[-2:] for a in x]
            img_list = models.detection.image_list.ImageList(x, img_shape)
            # visualize_and_save_images(x, targets) 
            
            # 如果 feature 是一个字典，请取消下面行的注释
            features = list(feature.values())
            # 如果 feature 是单个张量，请取消下面行的注释
            # features = [feature]
            
            # 确保 feature 是一个列表
            anchors = self.anchor_generator(img_list, features)  # 生成样本
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            
            
            # 对正负样本进行采样，以保持1:1比例
            positive_indices = [label == 1 for label in labels]
            negative_indices = [label == 0 for label in labels]
            
            # 对于每个图像，计算正样本的数量
            num_positives_per_image = [torch.sum(ind) for ind in positive_indices]

            # 对于每个图像，随机选择相应数量的负样本
            sampled_negative_indices = [
                torch.nonzero(ind).view(-1)[torch.randperm(torch.nonzero(ind).numel())[:num_positives_per_image[i]]]
                if torch.nonzero(ind).numel() > num_positives_per_image[i] else torch.nonzero(ind).view(-1)
                for i, ind in enumerate(negative_indices)
            ]

            # 提取对应的正样本标签
            sampled_positives = [labels[i][positive_indices[i]] for i in range(len(labels))]

            # 提取对应的负样本标签
            sampled_negatives = [labels[i][sampled_negative_indices[i]] for i in range(len(labels))]

            # 合并正负样本标签
            sampled_labels = [torch.cat((pos, neg), dim=0) for pos, neg in zip(sampled_positives, sampled_negatives)]

            # 将所有图像的正负样本标签连接起来
            concatenated_labels = torch.cat(sampled_labels, dim=0).to(torch.int64)
            
            # 提取对应的正样本锚点
            sampled_positive_anchors = [anchors[i][positive_indices[i]] for i in range(len(anchors))]

            # 提取对应的负样本锚点
            sampled_negative_anchors = [anchors[i][sampled_negative_indices[i]] for i in range(len(anchors))]

            # 合并正负样本锚点
            sampled_anchors = [torch.cat((pos, neg), dim=0) for pos, neg in zip(sampled_positive_anchors, sampled_negative_anchors)]
            # print(sum(anchor.numel() for anchor in sampled_anchors))
            # ... [continue with roi_head and loss calculation] ...

            # Now pass the concatenated_anchors to the roi_head
            # features_dict = feature = {"0": feature}
            features_dict = feature
            class_logits = self.roi_head(features_dict, sampled_anchors, tuple(img_list.image_sizes))
                      
            self.vis(x, anchors, sampled_positive_anchors, matched_gt_boxes, sampled_negative_anchors)
            # print("class_logits:", class_logits)
            
            loss = F.cross_entropy(class_logits, concatenated_labels, ignore_index=-1)            
            return loss
        
        elif self.mode == 'test':
            # 测试模式不变
            feature = self.backbone(x)
            img_shape = [a.shape[-2:] for a in x]
            img_list = models.detection.image_list.ImageList(x, img_shape)
            
            # 对于测试模式，确保 bbox 是正确格式
            bbox = [torch.tensor(targets).cuda().to(torch.float32)]
            # features_dict = {"0": feature}
            features_dict = feature
            class_logits = self.roi_head(features_dict, bbox, tuple(img_list.image_sizes))
            # visualize_and_save_images(x, list([[{'bbox':targets[0]}]]))
            # print("class_logits:", class_logits)
            
            return class_logits

    def assign_targets_to_anchors(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:

        labels = []
        matched_gt_boxes = []
        top_k = 200  # 假设我们想要选择的top-k个正样本

        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = torch.tensor([ann["bbox"] for ann in targets_per_image]).to('cuda')

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
                matched_idxs = self.proposal_matcher(match_quality_matrix)

                # 只选择top-k个正样本
                # 提取所有正样本的匹配质量分数
                positive_match_quality = match_quality_matrix.max(dim=0).values
                positive_idxs = matched_idxs >= 0

                # 使用正样本的匹配质量分数来选择top-k个
                _, topk_idxs = positive_match_quality[positive_idxs].topk(k=min(top_k, positive_idxs.sum()), largest=True)

                # 更新labels_per_image为-1，表示忽略
                labels_per_image = torch.full_like(matched_idxs, -1.0, dtype=torch.float32)

                # 设置top-k正样本的标签为1
                topk_anchors_idxs = torch.where(positive_idxs)[0][topk_idxs]
                labels_per_image[topk_anchors_idxs] = 1.0

                # 将背景anchors的标签设置为0
                labels_per_image[matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD] = 0.0

                # 获取每个top-k正样本对应的真实边界框
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]
                matched_gt_boxes_per_image = matched_gt_boxes_per_image[topk_anchors_idxs]

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes
        
    # def assign_targets_to_anchors(
    #     self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    # ) -> Tuple[List[Tensor], List[Tensor]]:

    #     labels = []
    #     matched_gt_boxes = []
    #     for anchors_per_image, targets_per_image in zip(anchors, targets):
    #         gt_boxes = torch.tensor([ann["bbox"] for ann in targets_per_image]).to('cuda')

    #         if gt_boxes.numel() == 0:
    #             # Background image (negative example)
    #             device = anchors_per_image.device
    #             matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
    #             labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
    #         else:
    #             match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image)
    #             matched_idxs = self.proposal_matcher(match_quality_matrix)
    #             # get the targets corresponding GT for each proposal
    #             # NB: need to clamp the indices because we can have a single
    #             # GT in the image, and matched_idxs can be -2, which goes
    #             # out of bounds
    #             matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

    #             labels_per_image = matched_idxs >= 0
    #             labels_per_image = labels_per_image.to(dtype=torch.float32)

    #             # Background (negative examples)
    #             bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
    #             labels_per_image[bg_indices] = 0.0

    #             # discard indices that are between thresholds
    #             inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
    #             labels_per_image[inds_to_discard] = -1.0

    #         labels.append(labels_per_image)
    #         matched_gt_boxes.append(matched_gt_boxes_per_image)
    #     return labels, matched_gt_boxes
        
class RoI_Head(nn.Module):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        out_channels,
        num_classes: int = 2,
        mode: str = 'train',
    ) -> None:
        super().__init__()
        if box_roi_pool is None:
            # self.box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
            self.box_roi_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
            

        if box_head is None:
            resolution = self.box_roi_pool.output_size[0]
            representation_size = 1024
            self.box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            self.box_predictor = nn.Linear(representation_size, num_classes)
            
    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits = self.box_predictor(box_features)
        bs = len(image_shapes)
        box_per_image = proposals[0].shape[0]
        
        return class_logits
            
            
class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super().__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x

def visualize_and_save_images(x, targets):
    import cv2
    import numpy
    # Convert the tensor to a numpy array and transpose it to [batch_size, height, width, channels]
    images = x.permute(0, 2, 3, 1).cpu().numpy().astype(numpy.uint8)
    
    for i, target in enumerate(targets):
        image = numpy.ascontiguousarray(images[i])
        # Extract the bounding box coordinates
        bbox = target[0]['bbox']  # Assuming bbox format is [x_min, y_min, width, height]
        x_min, y_min, x_max, y_max = bbox
        # x_max = x_min + width
        # y_max = y_min + height

        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        # Save the image with bounding box as a JPEG file
        cv2.imwrite(f'image_with_bbox_{i}.jpg', image)
