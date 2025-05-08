# CCR_kelp_feature_detection
This repo is intended to aggregate information regarding the testing of various feature detectors upon temperate, kelp forest benthic imagery.

A folder with 25 photos that exhibit a range of substrate and vegetation conditions for feature detector testing can be found [here](https://github.com/zhrandell/CCR_kelp_feature_detection/tree/main/photos). To provide a more robust testing scenario, we've including the following files for all 25 images (not that for a subset of the .GPR and .TIFF photos, we await the return of a staff member who is OOO and who has the original .GPRs locked in their Adobe account; all edited .JPEGs are present). 

- [unedited_GPR](https://github.com/Seattle-Aquarium/CCR_kelp_feature_detection/tree/main/photos/unedited_GPR) -- the "raw" photo file taken by GoPro; .GPR is a proprietary file type that we cannot write, but we can view and edit these files in Adobe Lightroom; maximum sensor information retained.
- [unedited_TIFF](https://github.com/Seattle-Aquarium/CCR_kelp_feature_detection/tree/main/photos/unedited_TIFF) -- we converted the .GPR to .TIFF to provide an unedited file format that retains much (but not all) of the original sensor information contained within the .GPR file. 
- [edited_TIFF](https://github.com/Seattle-Aquarium/CCR_kelp_feature_detection/tree/main/photos/edited_TIFF) -- these were included the enable "pre" and "post" photo editing .TIFFs (though the file size is much larger).
- [edited_JPEG](https://github.com/Seattle-Aquarium/CCR_kelp_feature_detection/tree/main/photos/edited_JPEG) -- our "standard" file output for edited (color-corrected, denoise, etc.) photos that are then processed with AI/ML to extract percent-cover and abundance data. 

We've applied our trained AI/ML algorithm (via [CoralNet-Toolbox](https://github.com/Jordan-Pierce/CoralNet-Toolbox)) to these 25 images, with 100 uniformly distributed (224 x 224 pixel) percent-cover patches per image. The predicted annotations for 2500 patches were manually reviewed and any errors were corrected. The .csv output can be found [here](https://github.com/Seattle-Aquarium/CCR_kelp_feature_detection/blob/main/data_output/25_images_percent-cover.csv), and it includes the (pixel) location information for all percent-cover patches. See https://github.com/Seattle-Aquarium/CCR_kelp_feature_detection/issues/1 for more information about the creation of these percent-cover data. 

<p float="center">
  <img src="figs/100_patches.png" width="400" height="347" />
  <img src="figs/100_annotated_patches.png" width="400" height="347" />
 </p>

<p float="center">
  <img src="figs/frequency_histogram.png" width="800" height="486" />
 </p>

 

