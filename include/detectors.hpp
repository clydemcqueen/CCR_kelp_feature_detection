#pragma once

namespace wrapped
{

// Abstract base class
class DetectorBase
{
public:
    virtual ~DetectorBase() = default;

    virtual void detectAndCompute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) = 0;

    [[nodiscard]] virtual std::string getName() const
    {
        constexpr auto prefix = "Feature2D.";

        const std::string name = detector_->getDefaultName();
        return name.substr(strlen(prefix));
    }

protected:
    cv::Ptr<cv::Feature2D> detector_;
};

// These detectors can detect keypoints _and_ compute descriptors, so they are useful for photogrammetry.
class DetectAndCompute : public DetectorBase
{
public:
    void detectAndCompute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) override
    {
        detector_->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    }
};

class SIFT final : public DetectAndCompute
{
public:
    SIFT()
    {
        detector_ = cv::SIFT::create();
    }
};

class BRISK final : public DetectAndCompute
{
public:
    BRISK()
    {
        detector_ = cv::BRISK::create();
    }
};

class ORB final : public DetectAndCompute
{
public:
    ORB()
    {
        detector_ = cv::ORB::create();
    }
};

class AKAZE final : public DetectAndCompute
{
public:
    AKAZE()
    {
        detector_ = cv::AKAZE::create();
    }
};

// These detectors cannot compute a descriptor, so they are less useful (not useful?) for photogrammetry.
class DetectOnly : public DetectorBase
{
public:
    void detectAndCompute (const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat&) override
    {
        detector_->detect(image, keypoints);
    }
};

class MSER final : public DetectOnly
{
public:
    MSER()
    {
        detector_ = cv::MSER::create();
    }
};

class FAST final : public DetectOnly
{
public:
    FAST()
    {
        detector_ = cv::FastFeatureDetector::create();
    }
};

class SimpleBlobDetector final : public DetectOnly
{
public:
    SimpleBlobDetector()
    {
        detector_ = cv::SimpleBlobDetector::create();
    }
};

class AgastFeatureDetector final : public DetectOnly
{
public:
    AgastFeatureDetector()
    {
        detector_ = cv::AgastFeatureDetector::create();
    }
};

class GFTTDetector final : public DetectOnly
{
public:
    GFTTDetector()
    {
        detector_ = cv::GFTTDetector::create();
    }
};

}
