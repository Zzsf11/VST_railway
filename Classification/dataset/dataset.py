import os.path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

from torchvision.datasets import CocoDetection
from torchvision import datapoints


class CocoDatasetBbox(CocoDetection):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, transforms, transform, target_transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        if self.transforms is not None:
            bboxes = []
            labels = []
            for ann in target:
                bboxes.append(ann['bbox'])
                labels.append(ann['category_id'])
            ori_size = image.size
            new_size = (ori_size[1], ori_size[0])
            bboxes = datapoints.BoundingBox(
                bboxes,
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=new_size,
            )
            
            image, bboxes, labels = self.transforms.target_transform(image, bboxes, labels)
            
            for i, bbox in enumerate(bboxes):
                target[i]['bbox'] = bbox.tolist()
                target[i]['category_id'] = labels[i]

        return image, target