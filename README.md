Nano-ISML：Single vessel analysis method based on protein nanoparticles and image segmentation machine learning


## train

Start visdom first：

```
python3 -m visdom.server
```

Then use the following command to start the training:

```
# Train on gpu0 and save the visualization results on the classifier env of visdom

CUDA_VISIBLE_DEVICES=3 python3 train_red_2.py train --use-gpu --env=a-1
CUDA_VISIBLE_DEVICES=3 python3 train_green_2.py train --use-gpu --env=a-1


```



## test

```
CUDA_VISIBLE_DEVICES=0 python.exe train_red_2.py test
CUDA_VISIBLE_DEVICES=0 python.exe train_green_2.py test
```


##  Data format

 FTn Confocal Image（Red/magenta） ： Image {}_c1.png
 Vessel Confocal Image（Green）： Image {}_c2.png
 
 FTn Confocal Image Annotations： 1.1.png
 Vessel Confocal Image Annotations： 2.1.png
