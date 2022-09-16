# LiDAR2camera_self_check
![image](https://github.com/OpenCalib/LiDAR2camera_self-check/blob/master/pictures/pipline.pdf)
![image](https://github.com/OpenCalib/LiDAR2camera_self-check/blob/master/pictures/input.pdf)
![image](https://github.com/OpenCalib/LiDAR2camera_self-check/blob/master/pictures/performance.pdf)
## Table of Contents

- [Requirements](#Requirements)
- [Evaluation](#Evaluation)
- [Train](#Train)
- [Citation](#Citation)

## Requirements

* python 3.6 (recommend to use [Anaconda](https://www.anaconda.com/))
* PyTorch==1.0.1.post2
* Torchvision==0.2.2
* Install requirements and dependencies
```commandline
pip install -r requirements.txt
```

## Evaluation

1. Download [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).
2. Change the path to the dataset in `evaluate_calib.py`.
```python
data_folder = '/path/to/the/KITTI/odometry_color/'
```
3. Create a folder named `pretrained` to store the pre-trained models in the root path.
4. Download pre-trained models and modify the weights path in `evaluate_calib.py`.
```python
weights = [
    # './pretrained/final_checkpoint_r20.00_t1.50_e4_0.094.tar',
    # './pretrained/final_checkpoint_r2.00_t0.20_e4_0.228.tar',
    # './pretrained/final_checkpoint_r10.00_t1.00_e3_0.108.tar',
    './pretrained/final_checkpoint_r5.00_t0.50_e-1_0.145.tar',
]
```
5. Run evaluation.
```commandline
python evaluate_calib.py
```

## Train
```commandline
python train_with_sacred.py
```



### Acknowledgments
 We are grateful to Daniele Cattaneo for his CMRNet [github repository](https://github.com/cattaneod/CMRNet) and LCCNet [github repository](https://github.com/IIPCVLAB/LCCNet). We use them as our initial code base.
 
<!-- [correlation_package](models/LCCNet/correlation_package) was taken from [flownet2](https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package)

[LCCNet.py](model/LCCNet.py) is a modified version of the original [PWC-DC network](https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py) and modified version [CMRNet](https://github.com/cattaneod/CMRNet/blob/master/models/CMRNet/CMRNet.py)  -->

---
