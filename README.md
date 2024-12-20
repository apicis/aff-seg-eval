# Evaluation toolkit for affordance segmentation

This repository contains the code to evaluate affordance segmentation models using two performance measures: 
* Jaccard index measures how many pixels predicted as class a certain class are correct, among all pixels.
* $F^w_{\beta}$ associates a different weight to the prediction errors based on the Euclidean
distance to the annotated mask.

[[arXiv](https://arxiv.org/abs/2409.01814)]
[[webpage](https://apicis.github.io/aff-seg/)]
[[models code](https://github.com/apicis/aff-seg)]
[[trained models](https://doi.org/10.5281/zenodo.13627870)]


## Table of Contents
1. [Installation](#installation)
    1. [Setup specifics](#setup_specifics)  
    2. [Requirements](#requirements)
    3. [Instructions](#instructions)
2. [Running demo](#demo)
3. [Contributing](#contributing)
4. [Credits](#credits)
5. [Enquiries, Question and Comments](#enquiries-question-and-comments)
6. [License](#license)


## Installation <a name="installation"></a>

### Setup specifics <a name="setup_specifics"></a>
The models testing were performed using the following setup:
* *OS:* Ubuntu 18.04.6 LTS
* *Kernel version:* 4.15.0-213-generic
* *CPU:* Intel® Core™ i7-9700K CPU @ 3.60GHz
* *Cores:* 8
* *RAM:* 32 GB
* *GPU:* NVIDIA GeForce RTX 2080 Ti
* *Driver version:* 510.108.03
* *CUDA version:* 11.6

### Requirements <a name="requirements"></a> 
* Python 3.8
* OpenCV 4.10.0.84
* Numpy 1.24.4
* Tqdm 4.66.5

### Instructions <a name="instructions"></a>
```
# Create and activate conda environment
conda create -n affordance_segmentation python=3.8
conda activate affordance_segmentation
    
# Install libraries
pip install opencv-python numpy tqdm scipy pandas scikit-learn
```

## Running demo <a name="demo"></a>

To run the evaluation toolkit and visualise the performance measure value (except for background):

```
python src/eval_toolkit.py --pred_dir=PRED_DIR --ann_dir=ANN_DIR --task=TASK --num_classes=NUM_CLASSES
```

* *PRED_DIR*: directory where predictions are stored
* *ANN_DIR*: directory where annotations are stored
* *TASK*: evaluation type: 1 for $F^w_{\beta}$, 2 for Jaccard index (IoU)
* *NUM_CLASSES*: number of output segmentation classes (background included)
* *SAVE_RES*: whether to save results or not
* *DEST_PATH*: path to destination .csv file (considered only if *SAVE_RES*=True)

You can evaluate also from the .csv file using `eval_from_file.py` script. We realease also [available models results](/assets/results) in .csv file. 


## Contributing <a name="contributing"></a>

If you find an error, if you want to suggest a new feature or a change, you can use the issues tab to raise an issue with the appropriate label.


## Credits <a name="credits"></a>

T. Apicella, A. Xompero, P. Gastaldo, A. Cavallaro, <i>Segmenting Object Affordances: Reproducibility and Sensitivity to Scale</i>, 
Proceedings of the European Conference on Computer Vision Workshops, Twelfth International Workshop on Assistive Computer Vision and Robotics (ACVR),
Milan, Italy, 29 September 2024.

```
@InProceedings{Apicella2024ACVR_ECCVW,
            title = {Segmenting Object Affordances: Reproducibility and Sensitivity to Scale},
            author = {Apicella, T. and Xompero, A. and Gastaldo, P. and Cavallaro, A.},
            booktitle = {Proceedings of the European Conference on Computer Vision Workshops},
            note = {Twelfth International Workshop on Assistive Computer Vision and Robotics},
            address={Milan, Italy},
            month="29" # SEP,
            year = {2024},
        }
```


## Enquiries, Question and Comments <a name="enquiries-question-and-comments"></a>

If you have any further enquiries, question, or comments, or you would like to file a bug report or a feature request, please use the Github issue tracker. 


## Licence <a name="license"></a>
This work is licensed under the MIT License.  To view a copy of this license, see [LICENSE](LICENSE).
