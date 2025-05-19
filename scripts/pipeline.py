# !/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


class Detection:
    """Results of a single call to detector.detect()"""

    def __init__(self, name: str, detector_name: str, keypoints: list):
        self.name = name                                                    # Name of the image or patch type
        self.detector_name = detector_name                                  # Name of the detector
        self.keypoints = keypoints                                          # Keypoints detected
        self.responses: list[float] = [kp.response for kp in keypoints]     # Just the response values

    @staticmethod
    def csv_header():
        return 'name,detector,num_features,r_min,r_max,r_mean,r_std\n'

    def csv_row(self):
        return f'{self.name},{self.detector_name},{len(self.responses)},{min(self.responses)},{max(self.responses)},{np.mean(self.responses)},{np.std(self.responses)}\n'


class FeaturePipeline:
    def __init__(self, detectors):
        self.detectors = detectors

        # Keep track of all detections, organized by detector
        self.detections_by_detector: dict[str, list[Detection]] = {}

    def process_image(self, image_path: str, output_dir: str, annotate: bool):
        print(f'Open {image_path}')
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'Failed to load image: {image_path}')
            return

        image_name = Path(image_path).stem
        partial_output_path = os.path.join(output_dir, f'{image_name}_')

        # Run each detector, collecting results
        for detector in self.detectors:
            detector_name = detector.__class__.__name__
            print(f'Start {detector_name}')

            if detector_name not in self.detections_by_detector:
                self.detections_by_detector[detector_name] = []

            keypoints = detector.detect(image, None)
            self.detections_by_detector[detector_name].append(Detection(image_name, detector_name, keypoints))

            if annotate:
                visualization = cv2.drawKeypoints(image, keypoints, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.imwrite(f'{partial_output_path}{detector_name}.jpg', visualization)

    def write_results(self, output_dir: str):
        stats_file = open(os.path.join(output_dir, 'stats_by_image.csv'), 'w')
        stats_file.write(Detection.csv_header())

        for detector_name, detections in self.detections_by_detector.items():
            combined_keypoints = []  # Collect all keypoints for this detector

            for detection in detections:
                combined_keypoints.extend(detection.keypoints)
                stats_file.write(detection.csv_row())

            # Write aggregate stats
            combined_detection = Detection(f'all', detector_name, combined_keypoints)
            stats_file.write(combined_detection.csv_row())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--detector', default='desc', help='detector type, choose from SIFT, BRISK, ORB, MSER, AKAZE, FAST, blob, Agast, GFTT, desc (default), all')
    parser.add_argument('-a', '--annotate', action='store_true', help='draw features on image')
    parser.add_argument('input', help='path to image or directory with images')
    parser.add_argument('output', help='output directory')
    args = parser.parse_args()
    
    if not os.path.isdir(args.output):
        print(f'Output path is not a directory: {args.output}')
        sys.exit(1)

    detectors = []

    # Feature detectors that can extract descriptors
    if args.detector in ['SIFT', 'desc', 'all']:
        detectors.append(cv2.SIFT_create())
    if args.detector in ['BRISK', 'desc', 'all']:
        detectors.append(cv2.BRISK_create())
    if args.detector in ['ORB', 'desc', 'all']:
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

    if os.path.isdir(args.input):
        for entry in os.scandir(args.input):
            if entry.name.endswith(('.jpg', '.png')):
                pipeline.process_image(entry.path, args.output, args.annotate)
    else:
        pipeline.process_image(args.input, args.output, args.annotate)

    pipeline.write_results(args.output)


if __name__ == '__main__':
    main()