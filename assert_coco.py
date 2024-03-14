"""
file : assert_coco.py

author : Aung Paing
cdate : Monday October 2nd 2023
mdate : Monday October 2nd 2023
copyright: 2023 GlobalWalkers.inc. All rights reserved.
"""
from collections import defaultdict


from coco.coco import COCO, AssertCOCO

anno_file = f"/home/aung/Documents/tmp/join.json"
img_dir = ""

assert_iou = False


def main():
    coco = COCO(anno_file)
    print(coco)

    coco_assert = AssertCOCO(coco)
    if img_dir:
        coco_assert.assert_img_level_annotations(img_dir, assert_iou)
    coco_assert.assert_anno_level_annotations()

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
