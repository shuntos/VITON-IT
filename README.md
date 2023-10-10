<h1 align="center">
  VTON-IT: Virtual Try-On using Image Translation
</h1>


<p align="center">
 This paper introduces VTON-IT, a novel Virtual Try-On application that uses semantic segmentation and a generative adversarial network to produce high-resolution, natural-looking images of clothes overlaid onto segmented body regions, addressing the challenges of body size, pose, and occlusions.
</p>


<div align="center">
  <a href="https://github.com/shuntos/VITON-IT/"><b>Project Page</b></a> |
  <a href="https://arxiv.org/pdf/2310.04558.pdf"><b>Paper</b></a> |
  <a href="[https://www.youtube.com/watch?v=nofH1K70tpc&t=30s](https://www.youtube.com/watch?v=sYdoLNQOzsk)"><b>Video</b></a>
</div>




<div align="center">
  <img src=final_overlay.jpg width="800">
</div>

## Requirements

- python 3.6.13
- torch 1.1.0 (as no third party libraries are required in this codebase, other versions should work, not yet tested)
- torchvision 0.3.0
- tensorboardX
- opencv

## Training Pix2pix:
```
 python3 train.py --label_nc 0 --no_instance --name vd2.0_2  --dataroot ./datasets/vd2.0_2 --continue_train   --gpu_ids 0,1 --batchSize 2 
```

## Train Segmentation model
```
u2net_train.py

```


## Inference
```
Inference.py
```

## Reference

If you find this repo helpful, please consider citing:

```
@misc{adhikari2023vtonit,
      title={VTON-IT: Virtual Try-On using Image Translation}, 
      author={Santosh Adhikari and Bishnu Bhusal and Prashant Ghimire and Anil Shrestha},
      year={2023},
      eprint={2310.04558},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
 The authors would like to thank IKebana Solutions LLC for providing them with constant support for this research project.






