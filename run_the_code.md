> TODO merge this with README.md when the dust settles

Detect features and generate descriptors if possible.

Tested with OpenCV 4.5.4 on Ubuntu 22.04.

There are 4 detectors that can generate descriptors:
- SIFT
- BRISK
- ORB
- AKAZE

There are 5 detectors that cannot generate descriptors:
- MSER
- FAST
- blob
- Agast
- GFTT

To install and build:
~~~
cd ~/projects
git clone git@github.com:Seattle-Aquarium/CCR_kelp_feature_detection.git
cd CCR_kelp_feature_detection
mkdir build
cd build
cmake ..
make
~~~

To run the 4 descriptor-generating detectors against a single edited jpeg:
~~~
cd ~/projects/CCR_kelp_feature_detection/build
mkdir temp_results
./opencv_pipeline desc ../photos/edited_JPEG/image_1.jpg temp_results
~~~

You'll find the results in `build/temp_results`:
~~~
~~~