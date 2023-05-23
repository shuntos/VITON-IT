# Demo video 

https://www.youtube.com/watch?v=nofH1K70tpc&t=30s

![Alt Text](final_overlay.jpg)


# VITON-IT
Virtual-Try On using Image Translation Repo


# Training Pix2pix:
```
 python3 train.py --label_nc 0 --no_instance --name vd2.0_2  --dataroot ./datasets/vd2.0_2 --continue_train   --gpu_ids 0,1 --batchSize 2 
```

# Train Sementation model
```
u2net_train.py

```


# Inference
```
Inference.py
```


