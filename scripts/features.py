# !/usr/bin/env python3

"""
Read the image patches, count features, write results.

All hard-coded:
    Read images from data_output/image_patches_from_25_test_photos/train/
    Runs 4 detectors: SIFT, BRISK, ORB, and AKAZE
    Write results to pipeline_results/features_by_detector.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2

from util import get_detectors


class SimplePipeline:
    def __init__(self, detectors):
        self.detectors = detectors
        self.results_file = open(os.path.join('../pipeline_results/features_by_detector.csv'), 'w')
        self.write_csv_header()

    def write_csv_header(self):
        self.results_file.write('image,patch')
        for detector in self.detectors:
            self.results_file.write(f',{detector.__class__.__name__}')
        self.results_file.write('\n')

    def write_csv_row(self, path: Path, feature_counts: list[int]):
        file_name = path.name
        patch_name = path.parent.name
        self.results_file.write(f'{file_name},{patch_name}')
        for feature_count in feature_counts:
            self.results_file.write(f',{feature_count}')
        self.results_file.write('\n')

    def run(self):
        entries = os.scandir('../data_output/image_patches_from_25_test_photos/train/')
        for entry in entries:
            if entry.is_dir():
                self.process_directory(entry.path)

    def process_directory(self, path: str):
        entries = os.scandir(path)
        for entry in entries:
            if entry.is_file() and entry.name.lower().endswith('.jpg'):
                    self.process_image(entry.path)

    def process_image(self, image_path: str):
        print(f'Open {image_path}')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'Failed to load image: {image_path}')
            return

        feature_counts: list[int] = []
        for detector in self.detectors:
            feature_counts.append(len(detector.detect(image, None)))

        self.write_csv_row(Path(image_path), feature_counts)


def main():
    pipeline = SimplePipeline(get_detectors('desc'))
    pipeline.run()


if __name__ == '__main__':
    main()