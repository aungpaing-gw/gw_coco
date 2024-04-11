"""
file : COCO.py

author : Aung Paing
cdate : Wednesday September 27th 2023
mdate : Wednesday September 27th 2023
copyright: 2023 GlobalWalkers.inc. All rights reserved.
"""
import json
import os.path as osp
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Union

import cv2

from coco.utils import assert_file, BoundingBox


class PersonKeypoints:
    def __init__(self) -> None:
        self.head_keypoints = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
        ]
        self.body_keypoints = [
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        self.keypoints = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        self.kp_colors = [
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
            (255, 0, 0),  # Red
        ]


class VisPersonSkeleton:
    def __init__(self):
        keypoints_obj = PersonKeypoints()
        self.person_keypoints = keypoints_obj.keypoints
        self.person_kps_color = keypoints_obj.kp_colors
        self.head_keypoints = keypoints_obj.head_keypoints
        self.body_keypoints = keypoints_obj.body_keypoints

        self.SUPER_CATEGORY = "person"
        self.person_skeleton = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

        self.vis_skeleton = {
            "left": [
                [1, 3],
                [1, 0],
                [0, 5],
                [5, 7],
                [7, 9],
                [5, 11],
                [11, 13],
                [13, 15],
            ],
            "right": [
                [4, 2],
                [0, 2],
                [0, 6],
                [6, 8],
                [8, 10],
                [6, 12],
                [12, 14],
                [16, 14],
            ],
        }
        self.vis_skeleton_color = {
            "left": [
                [0, 255, 0],
                [0, 255, 0],
                [0, 255, 0],
                [0, 255, 0],
                [0, 255, 0],
                [0, 255, 0],
                [0, 255, 0],
                [0, 255, 0],
            ],
            "right": [
                [0, 0, 255],
                [0, 0, 255],
                [0, 0, 255],
                [0, 0, 255],
                [0, 0, 255],
                [0, 0, 255],
                [0, 0, 255],
                [0, 0, 255],
            ],
        }


class COCO:
    """COCO Class for accessing coco dataset
    Implementation inspired from pycocotools

    Args:
        annotation_file (str): COCO format annotation file
    """

    def __init__(self, annotation_file: str):
        self.file_name = annotation_file
        self.data = self.read_json(self.file_name)
        self.cats = self._loadIndex(self.data["categories"])
        self.imgs = self._loadIndex(self.data["images"])
        self.annos = self._loadIndex(self.data["annotations"])

        self._img_anno, self._cat_img = self._get_image_annotation_pair()

    def __repr__(self):
        coco_str = "COCO Dataset format annotation\n"
        coco_str += f"Number of Categories \t : {len(self.cats)}\n"
        coco_str += f"Number of Image : \t : {len(self.imgs)}\n"
        coco_str += f"Number of Annotations \t : {len(self.annos)}"
        return coco_str

    @staticmethod
    def read_json(file_name: str):
        assert_file(file_name)
        return json.load(open(file_name, "r"))

    @staticmethod
    def _loadIndex(annotation_list_object: List[Dict]) -> Dict[int, Dict]:
        ret = {}
        for list_object in annotation_list_object:
            ret[list_object["id"]] = list_object
        return ret

    def _get_image_annotation_pair(self):
        img_anno = defaultdict(list)
        cat_img = defaultdict(list)

        for annoId, annotation in self.annos.items():
            imgId = annotation["image_id"]
            catId = annotation["category_id"]

            img_anno[imgId].append(annoId)
            cat_img[catId].append(imgId)

        # Consider for case when multiple same object exists in a single image
        for k, v in cat_img.items():
            cat_img[k] = list(set(v))
        return img_anno, cat_img

    def getImgIds(self, catIds: Optional[List[int]] = None):
        imgIds = []
        if catIds is None:
            imgIds = [imgId for imgId, _ in self.imgs.items()]
        else:
            imgIds = self._cat_img[catIds]
        return imgIds

    def getAnnIds(self, imgIds: Union[List[int], int]) -> List[int]:
        """Get all annotations ID for the given Image ID

        Args:
            imgIds (list[int]): imgID in the annotation
        Returns:
            annIds (list[int]): Annotation IDS for the imgIDs
        """
        imgIds = imgIds if isinstance(imgIds, list) else [imgIds]
        return [annoId for imgId in imgIds for annoId in self._img_anno[imgId]]

    def loadImgs(self, imgIds: Union[List[int], int]):
        imgIds = imgIds if isinstance(imgIds, list) else [imgIds]
        return [self.imgs[imgId] for imgId in imgIds]

    def loadAnns(self, annIds: Union[List[int], int]):
        annIds = annIds if isinstance(annIds, list) else [annIds]
        return [self.annos[annId] for annId in annIds]


class AssertCOCO:
    def __init__(self, coco: COCO):
        self.coco = coco

    def _assert_images(self, img_dir: str):
        """Assert the exist of image file name and the file is valid (readable)

        Args:
            img_dir (str): The base image dir name
        """
        assert_file(img_dir)
        for _, img in self.coco.imgs.items():
            img_base_name = img["file_name"]
            img_full_name = osp.join(img_dir, img_base_name)
            assert_file(img_full_name)
            cv2.imread(img_full_name)

    def _assert_annotations_iou(self, iou_threshold=0.5):
        """Assert the IOU of the annotation in the images"""
        for imgId, _ in self.coco.imgs.items():
            annIds = self.coco.getAnnIds(imgIds=imgId)
            img = self.coco.loadImgs(imgIds=[imgId])[0]
            img_base_name = img["file_name"]

            # TODO: Instead of using loop, use Array
            for i, annId in enumerate(annIds):
                anno = self.coco.loadAnns(annIds=annId)[0]
                bbox = BoundingBox(*anno["bbox"])
                for j, otherAnnId in enumerate(annIds[i + 1 :]):
                    otherAnno = self.coco.loadAnns(annIds=otherAnnId)[0]
                    otherBbox = BoundingBox(*otherAnno["bbox"])
                    iou = bbox.get_iou(otherBbox)
                    assert iou < iou_threshold, f"{img_base_name} has \
                        \nIOU : {iou},\nBBox : {bbox} && {otherBbox}"

    def assert_anno_level_annotations(self):
        """Assert the correctness of annotation in COCO format
        For each annotation assert
        - If image ID is in the image ID list.
        - If category ID is in the category ID list.
        - Bounding box annotation is positive and not out of image shape
        """
        # _imgIds = [imgId for imgId, _ in self.coco.imgs.items()]
        # _catIds = [catId for catId, _ in self.coco.cats.items()]
        # _annIds = [annId for annId, _ in self.coco.annos.items()]
        _duplicated_imgIds = self._get_duplicated_item(
            [_x["id"] for _x in self.coco.data["images"]]
        )
        _duplicated_catIds = self._get_duplicated_item(
            [_x["id"] for _x in self.coco.data["categories"]]
        )
        _duplicated_annIds = self._get_duplicated_item(
            [_x["id"] for _x in self.coco.data["annotations"]]
        )
        assert (
            len(_duplicated_imgIds) == 0
        ), f"Duplicated Image ID \t: {_duplicated_imgIds}"
        assert (
            len(_duplicated_catIds) == 0
        ), f"Duplicated Category ID \t: {_duplicated_catIds}"
        assert (
            len(_duplicated_annIds) == 0
        ), f"Duplicated Annotation ID \t: {_duplicated_annIds}"
        for _, anno in self.coco.annos.items():
            _imgId = anno["image_id"]
            _catId = anno["category_id"]

            # Assert bounding box correctness
            # Get current annotation bounding box coordinate
            _x1, _y1, _w, _h = [int(__x) for __x in anno["bbox"]]
            _x2, _y2 = _x1 + _w, _y1 + _h
            # Get image shape for this annotation
            _img = self.coco.imgs[_imgId]
            imgH, imgW = int(_img["height"]), int(_img["width"])
            _imgName = self.coco.loadImgs([_imgId])[0]["file_name"]

            assert (
                0 <= _x1 <= imgW
            ), f"bbox coordinate out of range: {_x1} / {imgW} in {_imgName}"
            assert (
                0 <= _x2 <= imgW
            ), f"bbox coordinate out of range: {_x2} / {imgW} in {_imgName}"
            assert (
                0 <= _y1 <= imgH
            ), f"bbox coordinate out of range: {_y1} / {imgH} in {_imgName}"
            assert (
                0 <= _y2 <= imgH
            ), f"bbox coordinate out of range: {_y2} / {imgH} in {_imgName}"

            assert self.coco.imgs.get(_imgId), f"Image ID :{_imgId} is not correct"
            assert self.coco.cats.get(_catId), f"Category ID :{_catId} is not correct"

    def assert_img_level_annotations(self, img_dir: str, assert_iou: bool):
        """Assert the correctness of annotation in image level
        For each image assert
        - Image file Exists
        - Image file readable ( not broken image file )
        - Annotations in the image has IOU less than 0.6

        Args:
            img_dir (str): The base image dir name
        """
        self._assert_images(img_dir)
        if assert_iou:
            print("Assert the IOU scores")
            self._assert_annotations_iou(0.9)

    @staticmethod
    def _get_duplicated_item(ls: list):
        duplicated_list = list(filter(lambda x: x[1] > 1, list(Counter(ls).items())))
        duplicated_items = [x[0] for x in duplicated_list]
        return duplicated_items
