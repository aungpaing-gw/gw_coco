from typing import Union, List, Dict, Any
from collections import defaultdict


import numpy as np
import cv2

from typing import Optional, List
from coco.utils import assert_file, COLOR_PALETTE, BoundingBox, FolderPath
from coco.coco import COCO


class BaseVisualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def _draw_transparent_bg(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: Union[np.ndarray, List[int]] = [0, 255, 0],
        opacity: float = 0.5,
    ):
        if isinstance(color, list):
            color = np.array(color)
        cropped = img[y1:y2, x1:x2]
        cropped = (cropped * opacity) + (color * (1 - opacity))
        cropped = cropped.astype(np.uint8)
        img[y1:y2, x1:x2] = cropped
        return img

    @staticmethod
    def _draw_dotted_line(
        img: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: List[int],
        line_thickness: int,
    ):
        dot_space = int(line_thickness * 1.2)
        dot_ratio = int(line_thickness * 1.5)

        h, w = y2 - y1, x2 - x1

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, [0, 0], [w, h], 255, line_thickness * 2)

        # Crop center
        x_s = line_thickness
        y_s = line_thickness
        x_e = w - line_thickness
        y_e = h - line_thickness

        # Mask Lines
        x_l = np.arange(x_s, x_e, dot_space)
        y_l = np.arange(y_s, y_e, dot_space)
        for i in range(len(x_l) // dot_ratio):
            mask[:, x_l[i * dot_ratio] : x_l[i * dot_ratio + 1]] = 0
        for i in range(len(y_l) // dot_ratio):
            mask[y_l[i * dot_ratio] : y_l[i * dot_ratio + 1], :] = 0

        cropped = img[y1:y2, x1:x2]
        cropped[mask == 255] = color
        return img

    def _vis_txt(
        self,
        img: np.ndarray,
        x1: int,
        y1: int,
        txt: str,
        txtbgColor: Optional[List[int]] = None,
        txtSize: float = 0.5,
        txtThickness: int = 1,
    ):
        __text_size = cv2.getTextSize(txt, self.font, txtSize, txtThickness)[0]
        x1, y1 = x1, y1 - 2
        bbox_x1, bbox_y1 = x1, y1 - int(1.4 * __text_size[1])
        bbox_x2, bbox_y2 = x1 + __text_size[0] + 1, y1 - 1

        if bbox_x1 < 0:
            bbox_x2 += abs(bbox_x1)
            bbox_x1 = 0
        if bbox_y1 < 0:
            bbox_y2 += abs(bbox_y1)
            bbox_y1 = 0

        if txtbgColor:
            cv2.rectangle(img, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), txtbgColor, -1)

        # print(bbox_y1, bbox_y2, bbox_x1, bbox_x2)
        # print(img.shape)
        # print(img[bbox_y1:bbox_y2, bbox_x1:bbox_x2].shape)
        txt_color = (
            (0, 0, 0)
            if np.mean(img[bbox_y1:bbox_y2, bbox_x1:bbox_x2]) > 122
            else (255, 255, 255)
        )

        cv2.putText(
            img,
            txt,
            (x1, y1),
            self.font,
            txtSize,
            txt_color,
            txtThickness,
        )
        return img

    def _vis_bbox(
        self,
        img: np.ndarray,
        bbox: BoundingBox,
        bboxColor: List[int],
        bboxLineThickness: int = 2,
        fill: bool = False,
        transparent: bool = False,
        dottedLine: bool = False,
    ):
        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        if fill:
            cv2.rectangle(img, [x1, y1], [x2, y2], bboxColor, -1)
        if transparent:
            self._draw_transparent_bg(img, x1, y1, x2, y2, bboxColor, 0.5)
        if dottedLine:
            self._draw_dotted_line(img, x1, y1, x2, y2, bboxColor, bboxLineThickness)
        if not fill and not transparent and not dottedLine:
            cv2.rectangle(img, [x1, y1], [x2, y2], bboxColor, bboxLineThickness)
        return img


class COCOVis(BaseVisualizer):
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
        vis_bbox_dotted: bool = False,
        vis_bbox_fill: bool = False,
        vis_bbox_transparent: bool = False,
    ):
        super().__init__()
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
        self.txt_size = vis_txt_size
        self.txt_thickness = vis_txt_thickness
        self.vis_bbox_dotted = vis_bbox_dotted
        self.vis_bbox_fill = vis_bbox_fill
        self.vis_bbox_transparent = vis_bbox_transparent
        self.txt_height = cv2.getTextSize(
            "gw_coco", self.font, self.txt_size, self.txt_thickness
        )[0][1]

    def draw_bbox(
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
        _color: List[int] = self.COLOR_PALETTE[catId]
        _color = list(_color)

        # Draw BBox ( Options : Dotted Line, Transparent BG, Fill, Not Fill [ Default ])
        img_arr = self._vis_bbox(
            img_arr,
            bbox,
            _color,
            fill=self.vis_bbox_fill,
            transparent=self.vis_bbox_transparent,
            dottedLine=self.vis_bbox_dotted,
        )

        # Write Class Name
        text = self.__coco.cats[catId]["name"]
        if txt_append != "":
            text += txt_append

        x1, y1 = int(bbox.x1), int(bbox.y1)
        if not self.vis_txt_above_bbox:
            y1 += int(1.4 * self.txt_height)

        self._vis_txt(img_arr, x1, y1, text, _color, self.txt_size, self.txt_thickness)

        return img_arr

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

    def draw_sample_count(self, img_arr: np.ndarray, annIds: List[int]):
        vis_img_arr = img_arr
        sample_counts = defaultdict(int)
        for annId in annIds:
            anno = self.__coco.loadAnns(annIds=[annId])[0]
            catId = anno["category_id"]
            sample_counts[catId] += 1

        H, W = vis_img_arr.shape[:2]
        txt_box_x, txt_box_y = W - 200, 0
        txt_height = cv2.getTextSize(
            "test", self.font, self.txt_size, self.txt_thickness
        )[0][1]
        txt_height = (txt_height * len(sample_counts)) + (len(sample_counts) * 10)

        cv2.rectangle(
            vis_img_arr, [txt_box_x, txt_box_y], [W, txt_height], [255, 255, 255], -1
        )
        for catId, counts in sample_counts.items():
            catName = self.__coco.cats[catId]["name"]
            txt = f"{catName:<10}: {counts}"
            __text_size = cv2.getTextSize(
                txt, self.font, self.txt_size, self.txt_thickness
            )[0]
            cv2.putText(
                vis_img_arr,
                txt,
                [txt_box_x, txt_box_y + int(1.4 * __text_size[1])],
                self.font,
                self.txt_size,
                [0, 0, 0],
                self.txt_thickness,
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
        annIds: List[int] = self.__coco.getAnnIds(imgIds=imgId)
        img = self.__coco.loadImgs(imgIds=[imgId])[0]
        img_base_name = img["file_name"]
        # img_full_name = osp.join(self.img_dir, img_base_name)
        img_full_name = self.folder_path[img_base_name]
        assert_file(img_full_name)
        img_arr: np.ndarray = cv2.imread(img_full_name)

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
                img_arr = self.draw_bbox(img_arr, bbox, catId, txt_append)
            if self.vis_kps:
                if anno.get("keypoints"):
                    kps = anno["keypoints"]
                    img_arr = self._vis_kps(img_arr, kps)
                else:
                    print(
                        f"Keypoint Info not exists in {img_full_name} and `vis_kps` is set to `True`"
                    )
        if self.vis_sample_count:
            img_arr = self.draw_sample_count(img_arr, annIds)
        return img_arr
