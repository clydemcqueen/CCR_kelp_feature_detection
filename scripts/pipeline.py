# !/usr/bin/env python3

"""
Read images, run feature detectors, and gather per-detector statistics.

Typical usage is to run this script recursively (-r) on a directory. Statistics will be aggregated at each directory.
For example, you can structure the images by patch type:

    train
        BR_encrust
            BR_encrust_foo.jpg
            BR_encrust_bar.jpg
            ...
            stats.csv -- covers BR_encrust/*.jpg
        BR_fucus
            BR_fucus_foo.jpg
            BR_fucus_bar.jpg
            ...
            stats.csv -- covers BR_encrust/*.jpg
        stats.csv -- covers train/BR_encrust/*.jpg and train/BR_fucus/*.jpg

Re-run this script on these directories to generate stats.csv for all images and patches:
python scripts/pipeline.py -d desc -r photos/edited_JPEG/
python scripts/pipeline.py -d desc -r data_output/image_patches_from_25_test_photos/train/

"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def output_path(input_path: str, suffix: str, ext: str) -> str:
    """Given foo/fee.fie, return foo/fee_suffix.ext."""
    path = Path(input_path)
    return str(path.parent / f"{path.stem}_{suffix}.{ext}")


def csv_header():
    """
    Columns:
        path            File path, ** indicates this is a directory
        detector        Name of the detector
        d_num           Number of detections (individual files processed)
        f_mean          Mean number of features per detection
        r_min           Minimum response value
        r_max           Maximum response value
        r_mean          Mean response value
        r_std           Standard deviation of response values
    """
    return 'path,detector,d_num,f_mean,r_min,r_max,r_mean,r_std\n'


class Detection:
    """Results of a single call to detector.detect()."""

    def __init__(self, path: str, detector_name: str, responses: list[float]):
        self.path = path                            # Image path
        self.detector_name = detector_name          # Name of the detector
        self.responses = responses                  # The keypoint response values

    def csv_row(self):
        if len(self.responses) > 0:
            d_num = 1
            f_mean = len(self.responses)
            r_min = min(self.responses)
            r_max = max(self.responses)
            r_mean = np.mean(self.responses)
            r_std = np.std(self.responses)
            return f'{self.path},{self.detector_name},{d_num},{f_mean},{r_min},{r_max},{r_mean},{r_std}\n'
        else:
            return f'{self.path},{self.detector_name},0,0,0,0,0,0\n'


class DetectionList:
    """A list of detections from one detector."""

    def __init__(self, path: str, detector_name: str):
        self.path = os.path.join(path, '**')  # Special file name '**'
        self.detector_name = detector_name
        self.responses = []
        self.num_detections = 0

    def add(self, detection: Detection | DetectionList):
        assert detection.detector_name == self.detector_name
        self.responses.extend(detection.responses)
        if isinstance(detection, DetectionList):
            self.num_detections += detection.num_detections
        else:
            self.num_detections += 1

    def csv_row(self):
        if self.num_detections > 0:
            d_num = self.num_detections
            f_mean = len(self.responses)/self.num_detections
            r_min = min(self.responses)
            r_max = max(self.responses)
            r_mean = np.mean(self.responses)
            r_std = np.std(self.responses)
            return f'{self.path},{self.detector_name},{d_num},{f_mean},{r_min},{r_max},{r_mean},{r_std}\n'
        else:
            return f'{self.path},{self.detector_name},0,0,0,0,0,0\n'


class FeaturePipeline:
    def __init__(self, detectors):
        self.detectors = detectors

    def process_directory(self, path: str, recurse: bool, annotate: bool) -> dict[str, DetectionList]:
        """
        Process a directory and its subdirectories and write stats.csv.

        Image stats include the full path:
        path/to/image.jpg, detector_name, stats...

        Overall directory stats use the special name '**':
        path/to/this/directory/**, detector_name, stats...
        """

        # Open the stats.csv file in this directory
        stats_file = open(os.path.join(path, 'stats.csv'), 'w')
        stats_file.write(csv_header())

        # Accumulate responses (by detector) so that we can generate summary stats for this directory
        detection_lists: dict[str, DetectionList] = {}
        for detector in self.detectors:
            detection_lists[detector.__class__.__name__] = DetectionList(path, detector.__class__.__name__)

        # Process all images and subdirectories
        entries = os.scandir(path)
        for entry in entries:

            if entry.is_file():
                if entry.name.lower().endswith('.jpg'):
                    detections = self.process_image(entry.path, annotate)

                    # We have a list of detections, one per detector
                    for detection in detections:
                        stats_file.write(detection.csv_row())
                        detection_lists[detection.detector_name].add(detection)

            elif entry.is_dir() and recurse:
                subdir_detection_lists = self.process_directory(entry.path, recurse, annotate)

                for detector_name, detection_list in subdir_detection_lists.items():
                    detection_lists[detector_name].add(detection_list)

        for detector_name, detection_list in detection_lists.items():
            stats_file.write(detection_list.csv_row())

        # Cascade everything upward
        return detection_lists

    def process_image(self, image_path: str, annotate: bool) -> list[Detection]:
        print(f'Open {image_path}')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'Failed to load image: {image_path}')
            return []

        # Run each detector, collecting results
        detections: list[Detection] = []
        for detector in self.detectors:
            detector_name = detector.__class__.__name__
            # print(f'Start {detector_name}')

            keypoints = detector.detect(image, None)
            detections.append(Detection(image_path, detector_name, [kp.response for kp in keypoints]))

            if annotate:
                visualization = cv2.drawKeypoints(image, keypoints, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imwrite(output_path(image_path, f'_{detector_name}', 'jpg'), visualization)

        return detections


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    parser.add_argument('-r', '--recurse', action='store_true', help='enter directories looking for images')
    parser.add_argument('-d', '--detector', default='desc', help='detector type, choose from SIFT, BRISK, ORB, MSER, AKAZE, FAST, blob, Agast, GFTT, desc (default), all')
    parser.add_argument('-a', '--annotate', action='store_true', help='draw features on images')
    parser.add_argument('path', help='a directory with images')
    args = parser.parse_args()
    
    if not os.path.isdir(args.path):
        print(f'path must be a directory: {args.path}')
        sys.exit(1)

    detectors = []

    # Feature detectors that can extract descriptors
    if args.detector in ['SIFT', 'desc', 'all']:
        detectors.append(cv2.SIFT_create())
    if args.detector in ['BRISK', 'desc', 'all']:
        detectors.append(cv2.BRISK_create())
    if args.detector in ['ORB', 'desc', 'all']:
        # Ask ORB for zillions of features to get "all"
        detectors.append(cv2.ORB_create(nfeatures=10000000))
    if args.detector in ['AKAZE', 'desc', 'all']:
        detectors.append(cv2.AKAZE_create())

    # Feature detectors that cannot extract descriptors
    if args.detector in ['MSER', 'all']:
        detectors.append(cv2.MSER_create())
    if args.detector in ['FAST', 'all']:
        detectors.append(cv2.FastFeatureDetector_create())
    if args.detector in ['SimpleBlobDetector', 'blob', 'all']:
        detectors.append(cv2.SimpleBlobDetector_create())
    if args.detector in ['AgastFeatureDetector', 'Agast', 'all']:
        detectors.append(cv2.AgastFeatureDetector_create())
    if args.detector in ['GFTTDetector', 'GFTT', 'all']:
        detectors.append(cv2.GFTTDetector_create())

    if not detectors:
        print(f'Unknown detector: {args.detector}')
        sys.exit(1)

    pipeline = FeaturePipeline(detectors)
    pipeline.process_directory(args.path, args.recurse, args.annotate)


if __name__ == '__main__':
    main()