"""
file : vis_coco.py

author : Aung Paing
cdate : Monday October 2nd 2023
mdate : Monday October 2nd 2023
copyright: 2023 GlobalWalkers.inc. All rights reserved.
"""
import os
import os.path as osp
from collections import defaultdict

import cv2

from coco.coco import COCO, COCOVis

anno_file = f"/home/aung/Documents/projects/pj_112/annotations/1_1_b_140.json"
img_dir = f"/home/aung/Documents/projects/pj_112/image/1_1_B"
dst_dir = f"/home/aung/Documents/projects/pj_112/image/vis"

os.makedirs(dst_dir, exist_ok=True)

vis_bbox = True
vis_kps = False
vis_txt_bg_color = True
vis_txt_above_bbox = True
vis_sample_count = True
vis_txt_attribute = ["Direction", "Occlusion"]
vis_txt_attribute_value = (
    True  # If True, will write value of attribute instead of attribute
)

sample_vis = 10


def main():
    coco = COCO(anno_file)
    print(coco)

    coco_vis = COCOVis(
        coco,
        img_dir,
        vis_bbox,
        vis_kps,
        vis_sample_count,
        vis_txt_bg_color,
        vis_txt_above_bbox,
        vis_txt_attribute,
        vis_txt_attribute_value,
    )

    imgIds = coco.getImgIds()
    imgIds = sorted(imgIds)

    idx = 0
    for imgId in imgIds:
        img = coco.loadImgs(imgIds=imgId)[0]
        img_base_name = img["file_name"]
        dst_full_name = osp.join(dst_dir, img_base_name)
        img_arr = coco_vis.vis(imgId)
        cv2.imwrite(dst_full_name, img_arr)
        idx += 1
        if idx == sample_vis:
            break

    # Get Total Sample counts
    sample_count = defaultdict(int)
    for _, anno in coco.annos.items():
        catId = anno["category_id"]
        sample_count[catId] += 1

    print("\nTotal Sample counts")
    ret = ""
    for catId, counts in sample_count.items():
        ret += f'{coco.cats[catId]["name"]:<14}: {counts}\n'
    print(ret)


if __name__ == "__main__":
    main()
