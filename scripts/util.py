import cv2

def get_detectors(detector: str) -> list:
    detectors = []
    
    # Feature detectors that can extract descriptors
    if detector in ['SIFT', 'desc', 'all']:
        detectors.append(cv2.SIFT_create())
    if detector in ['BRISK', 'desc', 'all']:
        detectors.append(cv2.BRISK_create())
    if detector in ['ORB', 'desc', 'all']:
        # Ask ORB for zillions of features to get "all"
        detectors.append(cv2.ORB_create(nfeatures=10000000))
    if detector in ['AKAZE', 'desc', 'all']:
        detectors.append(cv2.AKAZE_create())

    # Feature detectors that cannot extract descriptors
    if detector in ['MSER', 'all']:
        detectors.append(cv2.MSER_create())
    if detector in ['FAST', 'all']:
        detectors.append(cv2.FastFeatureDetector_create())
    if detector in ['SimpleBlobDetector', 'blob', 'all']:
        detectors.append(cv2.SimpleBlobDetector_create())
    if detector in ['AgastFeatureDetector', 'Agast', 'all']:
        detectors.append(cv2.AgastFeatureDetector_create())
    if detector in ['GFTTDetector', 'GFTT', 'all']:
        detectors.append(cv2.GFTTDetector_create())
    
    return detectors
