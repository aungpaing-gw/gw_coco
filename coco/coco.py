"""
file : COCO.py

author : Aung Paing
cdate : Wednesday September 27th 2023
mdate : Wednesday September 27th 2023
copyright: 2023 GlobalWalkers.inc. All rights reserved.
"""
import json
import os
import os.path as osp
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Union

import cv2
from cv2.gapi import wip
import numpy as np

COLOR_PALETTE = [
    (0, 0, 0),  # Black
    (255, 255, 255),  # White
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 128, 128),  # Gray
    (192, 192, 192),  # Silver
    (128, 0, 0),  # Maroon
    (128, 128, 0),  # Olive
    (0, 128, 0),  # Green (Dark)
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (0, 64, 128),  # Blue (Dark)
    (64, 0, 128),  # Purple (Dark)
    (128, 64, 0),  # Brown
    (0, 128, 64),  # Green (Light)
    (0, 64, 0),  # Green (Dark)
    (64, 0, 0),  # Red (Dark)
    (255, 128, 128),  # Light Red
    (255, 128, 0),  # Orange
    (255, 255, 128),  # Light Yellow
    (128, 255, 128),  # Light Green
    (128, 255, 255),  # Light Cyan
    (128, 128, 255),  # Light Blue
    (255, 128, 255),  # Light Purple
    (192, 128, 64),  # Tan
    (255, 192, 128),  # Peach
    (255, 192, 192),  # Light Pink
    (128, 128, 64),  # Olive (Dark)
    (128, 64, 64),  # Brown (Dark)
    (128, 64, 128),  # Purple (Medium)
    (128, 0, 64),  # Maroon (Dark)
    (64, 128, 0),  # Olive (Light)
    (192, 64, 0),  # Orange (Dark)
    (192, 192, 0),  # Yellow (Dark)
    (192, 192, 128),  # Khaki
    (192, 192, 64),  # Olive (Medium)
    (128, 192, 64),  # Olive (Medium-Dark)
    (192, 128, 128),  # Rose
    (64, 192, 128),  # Green (Medium-Light)
    (64, 192, 192),  # Teal (Medium)
    (64, 64, 192),  # Blue (Medium)
    (192, 64, 192),  # Purple (Medium-Light)
    (192, 192, 192),  # Light Gray
    (0, 0, 64),  # Blue (Dark)
    (0, 64, 64),  # Teal (Dark)
    (0, 0, 128),  # Blue (Dark)
    (0, 128, 128),  # Teal (Medium)
    (0, 64, 192),  # Blue (Medium-Light)
    (64, 0, 64),  # Purple (Dark)
    (64, 64, 0),  # Olive (Dark)
    (64, 0, 0),  # Maroon (Dark)
    (192, 0, 0),  # Red (Medium-Dark)
    (192, 0, 192),  # Purple (Medium-Dark)
    (192, 128, 0),  # Brown (Medium)
    (128, 128, 192),  # Blue (Medium-Light)
    (128, 0, 192),  # Purple (Medium-Dark)
    (192, 0, 128),  # Magenta (Dark)
    (128, 0, 64),  # Maroon (Medium-Dark)
    (128, 64, 192),  # Purple (Medium-Light)
    (64, 128, 192),  # Blue (Medium-Light)
    (192, 128, 192),  # Pink (Light)
    (192, 0, 64),  # Red (Medium-Dark)
    (192, 64, 128),  # Pink (Medium)
    (64, 192, 0),  # Green (Medium)
    (128, 192, 128),  # Green (Medium-Light)
    (128, 192, 192),  # Cyan (Medium)
    (192, 192, 128),  # Green (Medium-Light)
    (192, 192, 0),  # Yellow (Medium)
    (0, 192, 64),  # Green (Medium-Light)
    (0, 192, 192),  # Teal (Medium)
    (64, 192, 192),  # Teal (Medium-Light)
    (128, 128, 64),  # Olive (Medium-Dark)
    (0, 128, 192),  # Blue (Medium-Light)
    (192, 128, 64),  # Brown (Medium)
    (192, 128, 0),  # Brown (Medium)
    (128, 128, 0),  # Olive (Medium)
    (64, 128, 64),  # Olive (Medium)
    (192, 64, 64),  # Red (Medium-Dark)
    (0, 128, 64),  # Green (Medium)
    (64, 192, 0),  # Green (Medium)
    (128, 64, 64),  # Red (Medium-Dark)
    (64, 64, 192),  # Blue (Medium-Light)
    (128, 0, 192),  # Purple (Medium-Dark)
    (64, 192, 128),  # Teal (Medium-Light)
    (128, 192, 192),  # Cyan (Medium-Light)
]


def assert_file(file_name: str):
    assert osp.exists(file_name), f"{file_name} not exists"


class BoundingBox:
    def __init__(
        self,
        x1: Union[int, float],
        y1: Union[int, float],
        w: Union[int, float],
        h: Union[int, float],
    ):
        """COCO Format Bounding Box Class

        Args:
            x1 (Union[int, float]): Upper Left X coordinate
            y1 (Union[int, float]): Upper Left Y coordinate
            w (Union[int, float]): Width of Bounding Box
            h (Union[int, float]): Height of Bounding Box
        """
        self.__x1 = x1
        self.__y1 = y1
        self.__w = w
        self.__h = h
        self.__x2 = self.x1 + w
        self.__y2 = self.y1 + h
        self.__area = w * h

    def __repr__(self):
        return f"xyxy : [{self.x1:.2f},{self.y1:.2f},{self.x2:.2f},{self.y2:.2f}]"

    @property
    def x1(self) -> Union[int, float]:
        return self.__x1

    @property
    def x2(self) -> Union[int, float]:
        return self.__x2

    @property
    def y1(self) -> Union[int, float]:
        return self.__y1

    @property
    def y2(self) -> Union[int, float]:
        return self.__y2

    @property
    def w(self) -> Union[int, float]:
        return self.__w

    @property
    def h(self) -> Union[int, float]:
        return self.__h

    @property
    def area(self) -> Union[int, float]:
        return self.__area

    @property
    def xywh(self) -> List[Union[int, float]]:
        return [self.x1, self.y1, self.w, self.h]

    @property
    def xyxy(self) -> List[Union[int, float]]:
        return [self.x1, self.y1, self.x2, self.y2]

    def get_iou(self, other) -> float:
        _intersection_x1 = max(self.x1, other.x1)
        _intersection_y1 = max(self.y1, other.y1)
        _intersection_x2 = min(self.x2, other.x2)
        _intersection_y2 = min(self.y2, other.y2)

        _intersection = max(_intersection_x2 - _intersection_x1, 0) * max(
            _intersection_y2 - _intersection_y1, 0
        )
        _union = (self.area + other.area) - _intersection
        iou = _intersection / _union
        return iou


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


class FolderPath:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        assert_file(root_dir)
        self.file_index = self.__create_file_index()

    def __create_file_index(self):
        file_index = {}
        for root, _, files in os.walk(self.root_dir):
            if files:
                for file in files:
                    file_full_name = osp.join(root, file)
                    file_base_name = osp.basename(file)
                    assert_file(file_full_name)
                    file_index[file_base_name] = file_full_name
        return file_index

    def __getitem__(self, file_base_name: str):
        return self.file_index[file_base_name]


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
            # Read Other lanugage file name in Window
            cv2.imdecode(
                np.fromfile(img_full_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )

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


class COCOVis:
    def __init__(
        self,
        coco: COCO,
        img_dir: str,
        vis_bbox: bool = True,
        vis_kps: bool = False,
        vis_sample_count: bool = True,
        vis_txt_bg_color: bool = True,
        vis_txt_above_bbox: bool = False,
        vis_txt_attribute: List[str] = [],
        vis_txt_attribute_value: bool = False,
        COLOR_PALETTE: List[List[int]] = COLOR_PALETTE,
        vis_txt_size: float = 0.5,
        vis_txt_thickness: int = 1,
    ):
        self.__coco = coco
        self.img_dir = img_dir
        self.folder_path = FolderPath(self.img_dir)
        self.COLOR_PALETTE = COLOR_PALETTE
        self.vis_bbox = vis_bbox
        self.vis_kps = vis_kps
        self.vis_sample_count = vis_sample_count
        self.vis_txt_bg_color = vis_txt_bg_color
        self.vis_txt_above_bbox = vis_txt_above_bbox
        self.vis_txt_attribute = vis_txt_attribute
        self.vis_txt_attribute_value = vis_txt_attribute_value
        self.vis_person_skeleton = VisPersonSkeleton()
        self.__font = cv2.FONT_HERSHEY_SIMPLEX
        self.__txt_size = vis_txt_size
        self.__txt_thickness = vis_txt_thickness

    def _vis_bbox(
        self, img_arr: np.ndarray, bbox: BoundingBox, catId: int, txt_append: str = ""
    ) -> np.ndarray:
        """Visualizaion of the COCO format annotation

        Args:
            img_arr (np.ndarray): Numpy image array
            bbox (BoundingBox): Boundingbox with coordinate information
            catId (int): CategryID

        Returns:
            vis_img_arr: Visualization of the copy of original image
        """
        vis_img_arr = img_arr
        _color = self.COLOR_PALETTE[catId]
        _color_np = np.array(_color)

        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        # Draw Bounding Box Rectangle
        cv2.rectangle(vis_img_arr, [x1, y1], [x2, y2], _color, 2)

        # Write Class Name
        text = self.__coco.cats[catId]["name"]
        if txt_append != "":
            text += txt_append
        __text_size = cv2.getTextSize(
            text, self.__font, self.__txt_size, self.__txt_thickness
        )[0]

        txt_color = (0, 0, 0) if np.mean(_color_np) > 122 else (255, 255, 255)
        txt_bk_color = (_color_np * 0.7).astype(np.uint8).tolist()
        if self.vis_txt_above_bbox:
            txt_x1, txt_y1 = x1, y1 - 2
            txt_bbox_x1, txt_bbox_y1 = x1, y1 - int(1.4 * __text_size[1])
            txt_bbox_x2, txt_bbox_y2 = x1 + __text_size[0] + 1, y1 - 1
        else:
            txt_x1, txt_y1 = x1, y1 + __text_size[1]
            txt_bbox_x1, txt_bbox_y1 = x1, y1 + 1
            txt_bbox_x2, txt_bbox_y2 = (
                x1 + __text_size[0] + 1,
                y1 + int(1.4 * __text_size[1]),
            )

        if self.vis_txt_bg_color:
            cv2.rectangle(
                vis_img_arr,
                (txt_bbox_x1, txt_bbox_y1),
                (txt_bbox_x2, txt_bbox_y2),
                txt_bk_color,
                -1,
            )
        cv2.putText(
            vis_img_arr,
            text,
            (txt_x1, txt_y1),
            self.__font,
            self.__txt_size,
            txt_color,
            self.__txt_thickness,
        )
        return vis_img_arr

    def _vis_kps(self, img_arr: np.ndarray, kps: List[int]) -> np.ndarray:
        kps = np.array(kps)
        kpx = kps[0::3]
        kpy = kps[1::3]
        kpv = kps[2::3]
        # Vis Keypoints
        for kpIdx in range(len(kpv)):
            if kpv[kpIdx] != 0:
                _color = self.vis_person_skeleton.person_kps_color[kpIdx]
                cv2.circle(img_arr, [kpx[kpIdx], kpy[kpIdx]], 3, _color, -1)
        # Vis Keypoint Lines
        for body_part, kp_lines in self.vis_person_skeleton.vis_skeleton.items():
            kp_lines = np.array(kp_lines)
            for line_idx, kp_line in enumerate(kp_lines):
                if np.all(kpv[kp_line]) > 0:
                    _x1, _x2 = kpx[kp_line]
                    _y1, _y2 = kpy[kp_line]
                    _line_color = self.vis_person_skeleton.vis_skeleton_color[
                        body_part
                    ][line_idx]
                    cv2.line(img_arr, [_x1, _y1], [_x2, _y2], _line_color, 1)
        return img_arr

    def _vis_sample_count(self, img_arr: np.ndarray, annIds: List[int]):
        vis_img_arr = img_arr
        sample_counts = defaultdict(int)
        for annId in annIds:
            anno = self.__coco.loadAnns(annIds=[annId])[0]
            catId = anno["category_id"]
            sample_counts[catId] += 1

        H, W = vis_img_arr.shape[:2]
        txt_box_x, txt_box_y = W - 200, 0
        txt_height = cv2.getTextSize(
            "test", self.__font, self.__txt_size, self.__txt_thickness
        )[0][1]
        txt_height = (txt_height * len(sample_counts)) + (len(sample_counts) * 10)

        cv2.rectangle(
            vis_img_arr, [txt_box_x, txt_box_y], [W, txt_height], [255, 255, 255], -1
        )
        for catId, counts in sample_counts.items():
            catName = self.__coco.cats[catId]["name"]
            txt = f"{catName:<10}: {counts}"
            __text_size = cv2.getTextSize(
                txt, self.__font, self.__txt_size, self.__txt_thickness
            )[0]
            cv2.putText(
                vis_img_arr,
                txt,
                [txt_box_x, txt_box_y + int(1.4 * __text_size[1])],
                self.__font,
                self.__txt_size,
                [0, 0, 0],
                self.__txt_thickness,
            )

            txt_box_y += __text_size[1] + 10

        return vis_img_arr

    def vis(self, imgId: int):
        """Visualization of the COCO format for given image ID

        Args:
            imgId (_type_): _description_

        Returns:
            vis_img_arr: Visualization of the image
        """
        annIds = self.__coco.getAnnIds(imgIds=imgId)
        img = self.__coco.loadImgs(imgIds=[imgId])[0]
        img_base_name = img["file_name"]
        # img_full_name = osp.join(self.img_dir, img_base_name)
        img_full_name = self.folder_path[img_base_name]
        assert_file(img_full_name)
        img_arr = cv2.imdecode(
            np.fromfile(img_full_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED
        )

        for annId in annIds:
            anno = self.__coco.loadAnns(annIds=[annId])[0]
            bbox = BoundingBox(*anno["bbox"])
            catId = anno["category_id"]
            txt_append = ""
            if len(self.vis_txt_attribute):
                for anno_attribute in self.vis_txt_attribute:
                    if anno["attributes"].get(anno_attribute):
                        if self.vis_txt_attribute_value:
                            txt_append += f" {anno['attributes'].get(anno_attribute)}"
                        else:
                            txt_append += anno_attribute[:4]

            if self.vis_bbox:
                img_arr = self._vis_bbox(img_arr, bbox, catId, txt_append)
            if self.vis_kps:
                if anno.get("keypoints"):
                    kps = anno["keypoints"]
                    img_arr = self._vis_kps(img_arr, kps)
                else:
                    print(
                        f"Keypoint Info not exists in {img_full_name} and `vis_kps` is set to `True`"
                    )
        if self.vis_sample_count:
            img_arr = self._vis_sample_count(img_arr, annIds)
        return img_arr
