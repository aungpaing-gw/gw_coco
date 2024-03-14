"""
file : coco_report.py

author : Aung Paing
cdate : Monday September 18th 2023
mdate : Monday September 18th 2023
copyright: 2023 GlobalWalkers.inc. All rights reserved.
"""
import json
import os
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COCO_COLOR_PALETTES = np.array(
    [
        [
            0.000,
            0.447,
            0.741,
        ],
        [
            0.850,
            0.325,
            0.098,
        ],
        [
            0.929,
            0.694,
            0.125,
        ],
        [
            0.494,
            0.184,
            0.556,
        ],
        [
            0.466,
            0.674,
            0.188,
        ],
        [
            0.301,
            0.745,
            0.933,
        ],
        [
            0.635,
            0.078,
            0.184,
        ],
        [
            0.300,
            0.300,
            0.300,
        ],
        [
            0.600,
            0.600,
            0.600,
        ],
        [
            1.000,
            0.000,
            0.000,
        ],
        [
            1.000,
            0.500,
            0.000,
        ],
        [
            0.749,
            0.749,
            0.000,
        ],
        [
            0.000,
            1.000,
            0.000,
        ],
        [
            0.000,
            0.000,
            1.000,
        ],
        [
            0.667,
            0.000,
            1.000,
        ],
        [
            0.333,
            0.333,
            0.000,
        ],
        [
            0.333,
            0.667,
            0.000,
        ],
        [
            0.333,
            1.000,
            0.000,
        ],
        [
            0.667,
            0.333,
            0.000,
        ],
        [
            0.667,
            0.667,
            0.000,
        ],
        [
            0.667,
            1.000,
            0.000,
        ],
        [
            1.000,
            0.333,
            0.000,
        ],
        [
            1.000,
            0.667,
            0.000,
        ],
        [
            1.000,
            1.000,
            0.000,
        ],
        [
            0.000,
            0.333,
            0.500,
        ],
        [
            0.000,
            0.667,
            0.500,
        ],
        [
            0.000,
            1.000,
            0.500,
        ],
        [
            0.333,
            0.000,
            0.500,
        ],
        [
            0.333,
            0.333,
            0.500,
        ],
        [
            0.333,
            0.667,
            0.500,
        ],
        [
            0.333,
            1.000,
            0.500,
        ],
        [
            0.667,
            0.000,
            0.500,
        ],
        [
            0.667,
            0.333,
            0.500,
        ],
        [
            0.667,
            0.667,
            0.500,
        ],
        [
            0.667,
            1.000,
            0.500,
        ],
        [
            1.000,
            0.000,
            0.500,
        ],
        [
            1.000,
            0.333,
            0.500,
        ],
        [
            1.000,
            0.667,
            0.500,
        ],
        [
            1.000,
            1.000,
            0.500,
        ],
        [
            0.000,
            0.333,
            1.000,
        ],
        [
            0.000,
            0.667,
            1.000,
        ],
        [
            0.000,
            1.000,
            1.000,
        ],
        [
            0.333,
            0.000,
            1.000,
        ],
        [
            0.333,
            0.333,
            1.000,
        ],
        [
            0.333,
            0.667,
            1.000,
        ],
        [
            0.333,
            1.000,
            1.000,
        ],
        [
            0.667,
            0.000,
            1.000,
        ],
        [
            0.667,
            0.333,
            1.000,
        ],
        [
            0.667,
            0.667,
            1.000,
        ],
        [
            0.667,
            1.000,
            1.000,
        ],
        [
            1.000,
            0.000,
            1.000,
        ],
        [
            1.000,
            0.333,
            1.000,
        ],
        [
            1.000,
            0.667,
            1.000,
        ],
        [
            0.333,
            0.000,
            0.000,
        ],
        [
            0.500,
            0.000,
            0.000,
        ],
        [
            0.667,
            0.000,
            0.000,
        ],
        [
            0.833,
            0.000,
            0.000,
        ],
        [
            1.000,
            0.000,
            0.000,
        ],
        [
            0.000,
            0.167,
            0.000,
        ],
        [
            0.000,
            0.333,
            0.000,
        ],
        [
            0.000,
            0.500,
            0.000,
        ],
        [
            0.000,
            0.667,
            0.000,
        ],
        [
            0.000,
            0.833,
            0.000,
        ],
        [
            0.000,
            1.000,
            0.000,
        ],
        [
            0.000,
            0.000,
            0.167,
        ],
        [
            0.000,
            0.000,
            0.333,
        ],
        [
            0.000,
            0.000,
            0.500,
        ],
        [
            0.000,
            0.000,
            0.667,
        ],
        [
            0.000,
            0.000,
            0.833,
        ],
        [
            0.000,
            0.000,
            1.000,
        ],
        [
            0.000,
            0.000,
            0.000,
        ],
        [
            0.143,
            0.143,
            0.143,
        ],
        [
            0.286,
            0.286,
            0.286,
        ],
        [
            0.429,
            0.429,
            0.429,
        ],
        [
            0.571,
            0.571,
            0.571,
        ],
        [
            0.714,
            0.714,
            0.714,
        ],
        [
            0.857,
            0.857,
            0.857,
        ],
        [
            0.000,
            0.447,
            0.741,
        ],
        [
            0.314,
            0.717,
            0.741,
        ],
        [0.50, 0.5, 0],
        [0.0, 0.0, 0],
        [
            0.333,
            1.000,
            1.000,
        ],
        [
            0.667,
            0.000,
            1.000,
        ],
        [
            0.667,
            0.333,
            1.000,
        ],
        [
            0.667,
            0.667,
            1.000,
        ],
        [
            0.667,
            1.000,
            1.000,
        ],
        [
            1.000,
            0.000,
            1.000,
        ],
        [
            1.000,
            0.333,
            1.000,
        ],
        [
            1.000,
            0.667,
            1.000,
        ],
        [
            0.333,
            0.000,
            0.000,
        ],
        [
            0.500,
            0.000,
            0.000,
        ],
        [
            0.667,
            0.000,
            0.000,
        ],
    ]
)


def assert_file(file_name: str):
    assert osp.exists(file_name), f"{file_name} not exists"


class COCOReport:
    def __init__(self, anno_file: str):
        assert_file(anno_file)
        self.anno_file = anno_file
        self.data = self._read_json()
        self.coco_stats = COCOStats(self.data)
        self.color_palette = self._get_color_palette()

        self.heatmap = HeatMap(self.coco_stats, self.color_palette)

        self.num_classes = len(self.coco_stats._categories)
        self.figure_size = self._get_figure_size()

        # Get Area Distribution
        self.area_fig, self.area_ax = self._get_area_distribution()
        # Get class Count
        self.cls_distribution_fig, self.cls_distribution_ax = self._get_class_count()

    def _read_json(self):
        return json.load(open(self.anno_file, "r"))

    def _get_color_palette(self):
        color_palette = {}
        _tmp_df = self.coco_stats._df_categories.sort_index()
        idxs = _tmp_df.index
        for i in range(len(_tmp_df)):
            color_palette[idxs[i]] = COCO_COLOR_PALETTES[i]
        return color_palette

    def _get_figure_size(self):
        fig_size = 8
        if self.num_classes < 10:
            fig_size = 10
        if self.num_classes >= 10 and self.num_classes <= 40:
            fig_size = 14
        if self.num_classes > 40:
            fig_size = 20
        return fig_size

    def _initialize_figure(self):
        plt.rcParams.update({"font.size": self.figure_size})
        fig, ax = plt.subplots(figsize=(self.figure_size, self.figure_size))
        plt.subplots_adjust(bottom=0.20, left=0.25)
        return fig, ax

    def _get_area_distribution(self):
        fig, ax = self._initialize_figure()
        sns.violinplot(
            self.coco_stats._df_annotations,
            x="area",
            y="category_id",
            linewidth=self.figure_size * 0.08,
            width=self.figure_size * 0.2,
            palette=self.color_palette,
            ax=ax,
        )
        ax.set_yticks(
            ticks=range(len(self.coco_stats._df_categories)),
            labels=self.coco_stats._df_categories["name"],
        )
        ax.set_xticklabels(
            [int(x) for x in ax.get_xticks()],
            rotation=50,
        )
        return fig, ax

    def _get_class_count(self):
        class_count_stat = (
            self.coco_stats._df_annotations["category_id"].value_counts().sort_index()
        )
        fig, ax = self._initialize_figure()
        sns.barplot(
            x=class_count_stat.values,
            y=class_count_stat.index,
            palette=self.color_palette,
            ax=ax,
        )
        ax.set_xlabel("Category Occurance Count")
        ax.set_yticks(
            ticks=range(len(self.coco_stats._df_categories)),
            labels=self.coco_stats._df_categories["name"],
        )
        ax.set_xticklabels(
            [int(x) for x in ax.get_xticks()],
            rotation=50,
        )
        return fig, ax


class COCOStats:
    def __init__(self, coco_data: dict):
        self._coco_data = coco_data
        self._images = self._coco_data["images"]
        self._categories = self._coco_data["categories"]
        self._annotations = self._coco_data["annotations"]

        self._df_images = pd.DataFrame(self._images)
        self._df_categories = pd.DataFrame(self._categories)
        self._df_annotations = pd.DataFrame(self._annotations)

        self._df_images = self._set_index(self._df_images, "id")
        self._df_categories = self._set_index(self._df_categories, "id")
        self._df_annotations = self._set_index(self._df_annotations, "id")

        self._df_images = self._change_dtype(self._df_images, "height", "uint16")
        self._df_images = self._change_dtype(self._df_images, "width", "uint16")
        self._df_annotations = self._change_dtype(
            self._df_annotations, "category_id", "category"
        )
        self._df_annotations = self._change_dtype(
            self._df_annotations, "image_id", "uint32"
        )
        self._df_annotations = self._change_dtype(
            self._df_annotations, "iscrowd", "bool"
        )

        self.total_images = len(self._df_images)
        self.total_annotations = len(self._df_annotations)

    @staticmethod
    def _set_index(df: pd.DataFrame, attribute: str):
        if df.get(attribute) is not None:
            df = df.set_index(attribute)
        return df

    @staticmethod
    def _drop_column(df: pd.DataFrame, col_name: str):
        if df.get(col_name) is not None:
            df = df.drop(columns=[col_name])
        return df

    @staticmethod
    def _change_dtype(df: pd.DataFrame, attribute: str, dtype: str):
        df[attribute] = df[attribute].astype(dtype)
        return df


class HeatMap:
    def __init__(self, coco_stats: COCOStats, color_palette: dict):
        self._heatmap_point_size = 11
        self._center = self._heatmap_point_size // 2
        self.heatmap_height = 1080
        self.heatmap_width = 1920

        self.annotations = coco_stats._annotations
        self.images = coco_stats._df_images
        self._COLORS = COCO_COLOR_PALETTES
        self.color_palette = color_palette
        self.colors = self._get_color()

        # Get heatmap
        self.heatmap = self._get_heatmap()

    def _get_color(self):
        colors = {}
        for _id, _color in self.color_palette.items():
            _color = _color.tolist()
            _tmp = np.zeros(
                [self._heatmap_point_size, self._heatmap_point_size, 3],
                dtype=np.float64,
            )
            _tmp = cv2.circle(_tmp, (self._center, self._center), 5, _color, -1)
            colors[_id] = _tmp
        return colors

    def _get_heatmap(self):
        heatmap = np.zeros((self.heatmap_height, self.heatmap_width, 3))

        for annotation in self.annotations:
            img_id = annotation["image_id"]
            try:
                image = self.images.loc[[img_id]]
                H, W = image["height"].iloc[0], image["width"].iloc[0]
            except IndexError:
                H, W = 1080, 1920
                print(f"image with image ID of {img_id} not found.")
                continue
            x1, y1, w, h = annotation["bbox"]
            xc, yc = int(x1 + w // 2), int(y1 + h // 2)
            xc_norm = xc // W
            yc_norm = yc // H

            heatmap_x = int(self.heatmap_width * xc_norm)
            heatmap_y = int(self.heatmap_height * yc_norm)

            cls_id = annotation["category_id"]

            min_x = heatmap_x - self._center - 1
            min_y = heatmap_y - self._center - 1
            max_x = heatmap_x + self._center
            max_y = heatmap_y + self._center
            if (
                min_x < 0
                or min_y < 0
                or max_x > self.heatmap_width
                or max_y > self.heatmap_height
            ):
                continue
            try:
                heatmap[min_y:max_y, min_x:max_x, :] += self.colors[cls_id]
            except IndexError:
                print(f"Index out of range : {cls_id} / {len(self.colors)}")

        # Post process
        heatmap /= max(np.max(heatmap), 0.001)
        heatmap = heatmap**0.3
        heatmap *= 255
        heatmap = heatmap.astype(np.uint8)
        return heatmap
