# YoloV5 with Object tracking (DeepSORT)

## Introduction
This repository contains the source code of YoloV5 andd DeepSORT pytorch. YoloV5 is an object detection algorithm and DeepSORT is an object tracking algorithm. The source code has been collected from their respective official repositories and modified for custom object detection and tracking. 

If you want to use it, I am giving the instructions below. I hope you find it easier.

A big shout out to Ultralytics and  ZQPei for YoloV5 and DeepSORT Pytorch.
1. https://github.com/ultralytics/yolov5
2. https://github.com/ZQPei/deep_sort_pytorch




# Files

There are two repositories here, YoloV5 and DeepSORT. DeepSORT source code is in the deep_sort_pytorch folder. If you want to study DeepSORT, here is a video to start with. https://www.youtube.com/watch?v=LbyqsoLJu5Q. (Shout out to Augmented Startups)



## Training DeepSORT
#### - What do you want to track?
First you need to figure out what type of object you want to track. If you want to run YOLOv5 and DeepSORT together on your custom dataset then you have to train these algorithms individually. YoloV5 training process is already available on the official repository. So I will avoid that part.

For my purpose, I am tracking some pet bottles moving on a conveyor belt. So at first I need to train the deepSORT model with these bottle images. 

Here is a thing, how many types of bottle do I have? I have at least 10 types of bottles. YOLOv5 doesn't care about the types of the bottles. It just detects all the bottles as "Bottle". But to make your tracking algorithm better, it's necessary to identify the types and group them into sub-folders. Because deepSORT training requires them structured in a folder where each subfolder is a class directory. It's similar to ImageFolder functionality.


For my purpose, the folder structure of the dataset is like:
-     bottles
	* green
	* red
	* small
	* glass
	* ...
	
	
This structure is required to train deepSORT. If you can't make this kind of structure then just create one folder inside your dataset folder and paste all the images. It's more like classification problem where you have Sub-folders as classes. If you have two types of object you want to track then you can do the following.
-     1. dataset
	* cars
		- image 1
		- image 2
	* bottle
		- image 1
		- image 2
	
	Or,
-     2. dataset
	* red_cars
		- image 1
		- image 2
	* black_cars or whatever
		- image 1
		- image 2
	* small_bottles
		- ........
	* glass_bottles
		- ........

I have tried both of these structures. (2) does better tracking as it learns feature representation for each type. (1) is also fine. No worries.  

Jump into **deep_sort_pytorch > deep_sort > deep** folder. You will find the source code of training here. 

All you need is the **path of the dataset** and **train.py**

Command:
```
python train.py --data-dir path/to/dataset 
```
Just make sure to edit the **deep_sort.yaml** in the configs folder in case you are resuming or have some pretrained weights. Just define the path of the cpkt. You shouldn't have it as we are training from scratch. So for now it doesn't matter.

Once the training has been finished, you'll find the weights in **checkpoint** folder. You can use this weight for inference or to train again. In case of training again, --resume has to be true. 


Here is a thing, once you finish training and try testing this code will throw an error. When you create a model instance just pass the number of classes or by default it will have 751 or 2 neurons on the last layers. Be careful.

**refer to feature_extractor.py Line no. 11.**
```
self.net = Net(num_classes= number of classes)
```

Once you are done training deepSORT, just train YoloV5, follow the official training guide and then run detect_and_track.py, same as Yolov5's detect.py but requires few arguments of deepSORT. 
```
--deep_sort_weights path/to/deepSORT/weight/file
--config_deepsort path/to/deepsort/config/file
```
You are all set. ;)
