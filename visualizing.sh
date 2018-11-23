#!/bin/bash

if [ $# != 2 ];then
echo "Usage: $0 SOURCE_DIR TARGET_DIR"
exit 1;
fi

while read LINE
do
	python vanilla_backprop_unet.py --image_path $1/$LINE.jpg --mask_path $2/$LINE.png  
done	
