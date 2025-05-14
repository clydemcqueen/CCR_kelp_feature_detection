#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <opencv2/opencv.hpp>

#include "detectors.hpp"

class FeaturePipeline
{
public:
    explicit FeaturePipeline(std::vector<std::shared_ptr<wrapped::DetectorBase>> detectors) : detectors_(std::move(detectors))
    {
    }

    void process_image(const std::string& image_path, const std::string& output_dir) const
    {
        // Read and convert to grayscale
        std::cout << "Open " << image_path << std::endl;
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }

        for (const auto& detector : detectors_) {
            std::cout << "Start " << detector->getName() << std::endl;

            // Detect features
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            detector->detectAndCompute(image, keypoints, descriptors);
            std::cout << "Detected " << keypoints.size() << " features" << std::endl;

            // Output filenames will start with output_path + stem + feature_name
            std::string stem = std::filesystem::path(image_path).stem().string();
            std::string partial_output_path = output_dir + "/" + stem + "_" + detector->getName();

            // Write features to CSV
            save_features_to_csv(partial_output_path + "_keypoints.csv", keypoints);

            // Draw features
            cv::Mat visualization;
            cv::drawKeypoints(image, keypoints, visualization, cv::Scalar::all(-1),
                              cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::imwrite(partial_output_path + ".jpg", visualization);
        }
}

private:
    std::vector<std::shared_ptr<wrapped::DetectorBase>> detectors_;

    static void save_features_to_csv(const std::string& filepath, const std::vector<cv::KeyPoint>& keypoints)
    {
        std::ofstream file(filepath);
        file << "x,y,size,angle,response,octave\n";
        for (const auto& kp : keypoints) {
            file << kp.pt.x << "," << kp.pt.y << "," << kp.size << ","
                << kp.angle << "," << kp.response << "," << kp.octave << "\n";
        }
    }
};

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cout << "Usage: " << argv[0] << " <detector_type> <input_path> <output_path>\n";
        std::cout << "Possible detectors: SIFT, BRISK, ORB, MSER, AKAZE, FAST, blob, Agast, GFTT, desc, all\n";
        return 1;
    }

    std::string detector_type = argv[1];
    std::string input_path = argv[2];
    std::string output_dir = argv[3];

    if (!std::filesystem::is_directory(output_dir)) {
        std::cout << "Output path is not a directory: " << output_dir << std::endl;
        return 1;
    }

    std::vector<std::shared_ptr<wrapped::DetectorBase>> detectors = {};

    // Feature detectors that can extract descriptors
    if (detector_type == "SIFT" || detector_type == "desc" || detector_type == "all") {
        detectors.push_back(std::make_shared<wrapped::SIFT>());
    }
    if (detector_type == "BRISK" || detector_type == "desc" || detector_type == "all") {
        detectors.push_back(std::make_shared<wrapped::BRISK>());
    }
    if (detector_type == "ORB" || detector_type == "desc" || detector_type == "all") {
        detectors.push_back(std::make_shared<wrapped::ORB>());
    }
    if (detector_type == "AKAZE" || detector_type == "desc" || detector_type == "all") {
        detectors.push_back(std::make_shared<wrapped::AKAZE>());
    }

    // Feature detectors that cannot extract descriptors
    if (detector_type == "MSER" || detector_type == "all") {
        detectors.push_back(std::make_shared<wrapped::MSER>());
    }
    if (detector_type == "FAST" || detector_type == "all") {
        detectors.push_back(std::make_shared<wrapped::FAST>());
    }
    if (detector_type == "SimpleBlobDetector" || detector_type == "blob" || detector_type == "all") {
        detectors.push_back(std::make_shared<wrapped::SimpleBlobDetector>());
    }
    if (detector_type == "AgastFeatureDetector" || detector_type == "Agast" || detector_type == "all") {
        detectors.push_back(std::make_shared<wrapped::AgastFeatureDetector>());
    }
    if (detector_type == "GFTTDetector" || detector_type == "GFTT" || detector_type == "all") {
        detectors.push_back(std::make_shared<wrapped::GFTTDetector>());
    }

    if (detectors.empty()){
        std::cout << "Unknown detector type: " << detector_type << std::endl;
        return 1;
    }

    FeaturePipeline pipeline(detectors);

    if (std::filesystem::is_directory(input_path)) {
        for (const auto& entry : std::filesystem::directory_iterator(input_path)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
                pipeline.process_image(entry.path().string(), output_dir);
            }
        }
    }
    else {
        pipeline.process_image(input_path, output_dir);
    }

    return 0;
}
