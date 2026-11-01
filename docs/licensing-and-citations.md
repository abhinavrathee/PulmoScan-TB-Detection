# Licensing and citations

## Project code and documentation

The original PulmoScan source code, project-authored documentation, and logo in this repository are licensed under the [MIT License](../LICENSE). The copyright notice names Abhinav Rathee, the owner of the repository where this project is maintained.

The license is permissive so the project can be reused, adapted, and showcased in a portfolio. It applies only to original PulmoScan material. It does not grant rights to competition data, ImageNet images, downloaded pretrained weights, or third-party packages.

## Dataset and competition

The project materials identify the source as the Kaggle competition [Tuberculosis Detection from Chest X-Rays](https://www.kaggle.com/competitions/tuberculosis-detection-from-chest-x-rays), attributed in the supplied project overview to CDIS, IIT Kanpur. The competition page and its rules were not fully accessible during repository setup, and the Kaggle file listing exposed only a placeholder. I could not verify a specific dataset license or redistribution grant from the accessible source.

For that reason, this repository does **not** assign the dataset an MIT license. The competition terms and any original dataset-provider terms govern access, use, and redistribution. Confirm the exact rules accepted for the competition before publishing any images, labels, derived files, trained weights or submissions. The local `TB_Dataset/` directory is not covered by the project license even if it is present beside the source tree.

Suggested dataset citation (confirm the host attribution and data version against the competition page before formal publication):

> CDIS, Indian Institute of Technology Kanpur. *Tuberculosis Detection from Chest X-Rays* [Competition dataset]. Kaggle. https://www.kaggle.com/competitions/tuberculosis-detection-from-chest-x-rays

## EfficientNetB3 and pretrained weights

The training code calls `tf.keras.applications.EfficientNetB3(weights="imagenet")`. The [Keras EfficientNet documentation](https://keras.io/api/applications/efficientnet/efficientnet_models/) describes the architecture and ImageNet-pretrained option. The [Keras source repository](https://github.com/keras-team/keras/blob/master/LICENSE) licenses its software under Apache-2.0; that source-code license does not by itself establish a separate license for every downloaded weight file or for ImageNet images.

The pretrained weights are downloaded by Keras from its configured upstream model storage. This project does not bundle or redistribute those weight files. The [ImageNet site](https://www.image-net.org/about.php) explains that it does not own the copyrights in the images and that access to images is subject to its terms. The exact license applicable to the Keras-hosted weight artifact was not established from the API documentation reviewed here; users should check the current Keras and ImageNet terms before redistributing weights or relying on them for a use outside their permitted scope.

## Third-party software

TensorFlow/Keras, NumPy, pandas, and scikit-learn are installed as dependencies and are not relicensed by PulmoScan's MIT license. Their licenses and notices are supplied by their respective projects and distributions.

## Academic references

1. Tan, M. & Le, Q. V. (2019). [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html). *Proceedings of the 36th International Conference on Machine Learning*, 97, 6105–6114.
2. Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K. & Fei-Fei, L. (2009). [ImageNet: A Large-Scale Hierarchical Image Database](https://doi.org/10.1109/CVPR.2009.5206848). *2009 IEEE Conference on Computer Vision and Pattern Recognition*, 248–255.

The ImageNet paper citation is included because the code initializes the backbone with ImageNet weights. Follow the ImageNet site's current citation guidance and access terms for any formal research use.

## Software citation

GitHub can read the root [CITATION.cff](../CITATION.cff) file and present a citation for this repository. A plain-text reference is:

> Rathee, A. (2026). *PulmoScan: Chest X-ray Tuberculosis Classification* (0.1.0) [Computer software]. https://github.com/abhinavrathee/PulmoScan-TB-Detection
