# EPSNet: Efficient Panoptic Segmentation Network with Cross-layer Attention Fusion


This project hosts the code for implementing the EPSNet for panoptic segmentation.

 - [EPSNet: Efficient Panoptic Segmentation Network with Cross-layer Attention Fusion](https://arxiv.org/abs/2003.10142)

Some examples from our EPSNet model (19 fps on a 2080Ti and 38.9 PQ on COCO Panoptic test-dev):


![](https://i.imgur.com/wGbYWWI.png)
![](https://i.imgur.com/VEqaMRa.png)
![](https://i.imgur.com/CozJCfA.png)


## Models
Here are our EPSNet models trained on COCO Panoptic dataset along with their FPS on a 2080Ti and PQ on `val`:



| Image Size | Backbone      | FPS  | PQ  | Weights |
|:----------:|:-------------:|:----:|:----:|:----:|
| 550        | Resnet50-FPN | 20.6 | 35.8 | [Download](https://drive.google.com/file/d/1klQX2b9SSNnfmxPGoCBBgXxeybeX82yy/view?usp=sharing)
| 550        | Resnet101-FPN | 19.6 | 38.6  | [Download](https://drive.google.com/file/d/1pO1Vxy5tINr7YhZLfIsqjNerGhenx7o4/view?usp=sharing)


To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `epsnet` for `epsnet_resnet101_54_800000.pth`).


## Installation

This implementation is based on [Yolact](https://github.com/dbolya/yolact). Therefore the installation is the same as Yolact. 
- Clone this repository and enter it:
   ```Shell
   git clone https://github.com/neo85824/epsnet.git
   cd epsnet
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       ```
 - If you'd like to train EPSNet:
     -  Download the [COCO Panoptic dataset](https://cocodataset.org/#download) and the panoptic annotations  
     -  Place them like the folloing structures. Note that both `train` and `val` images are placed in the `coco/images` folder.
 
    ```
    data
    └───coco
    │   └───images
    │   │  │    000000022371.jpg
    │   │  └─── ...    
    │   └───annotations
    │   │  │    panoptic_val2017.json
    │   │  └─── ...     
    ```
 - If you'd like to evaluate EPSNet, please install the [COCO panoptic api](https://github.com/cocodataset/panopticapi).
   ```Shell
   pip install git+https://github.com/cocodataset/panopticapi.git

   ```


## Quantitative Results on COCO
```Shell
# Quantitatively evaluate a trained model on the entire validation set. Make sure you have COCO downloaded as above.
python panoptic_eval.py --trained_model=weights/epsnet_resnet101_54_800000.pth --score_threshold 0.2

```


## Images
```Shell
# Display qualitative results on the specified image.
python panopitc_eval.py --trained_model=weights/epsnet_resnet101_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.png

# Process an image and save it to another file.
python panopitc_eval.py --trained_model=weights/epsnet_resnet101_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input_image.png:output_image.png

# Process a whole folder of images.
python panopitc_eval.py --trained_model=weights/epsnet_resnet101_54_800000.pth --score_threshold=0.15 --top_k=15 --images=path/to/input/folder:path/to/output/folder
```
## Video
```Shell
# Display a video in real-time. "--video_multiframe" will process that many frames at once for improved performance.
# If you want, use "--display_fps" to draw the FPS directly on the frame.
python panopitc_eval.py --trained_model=weights/epsnet_resnet101_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=my_video.mp4

# Display a webcam feed in real-time. If you have multiple webcams pass the index of the webcam you want instead of 0.
python panopitc_eval.py --trained_model=weights/epsnet_resnet101_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=0

# Process a video and save it to another file. This uses the same pipeline as the ones above now, so it's fast!
python panopitc_eval.py --trained_model=weights/epsnet_resnet101_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=input_video.mp4:output_video.mp4
```
As you can tell, `eval.py` can do a ton of stuff. Run the `--help` command to see everything it can do.
```Shell
python panoptic_eval.py --help
```


# Training
By default, we train on COCO. Make sure to download the entire dataset using the commands above.
 - To train, grab an imagenet-pretrained model and put it in `./weights`.
   - For Resnet101, download `resnet101_reducedfc.pth` from [here](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
   - For Resnet50, download `resnet50-19c8e357.pth` from [here](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing).
 - Run one of the training commands below.
   - Note that you can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
   - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains using the base config with a batch size of 8 (the default).
python train.py --config=epsnet_resnet101_config

# Trains epsnet_resnet101_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
python train.py --config=epsnet_resnet101_config --batch_size=5

# Resume training epsnet with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=epsnet_resnet101_config --resume=weights/epsnet_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python train.py --help
```



# Citation
If you use EPSNet or this code base in your work, please cite
```
@article{EPSNet_arxiv,
  title={EPSNet: Efficient Panoptic Segmentation Network with Cross-layer Attention Fusion},
  author={Chia-Yuan Chang and Shuo-En Chang and P. Hsiao and L. Fu},
  journal={ArXiv},
  year={2020},
  volume={abs/2003.10142}
}
```


