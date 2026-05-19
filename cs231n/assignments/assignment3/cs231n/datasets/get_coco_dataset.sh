#!/bin/bash
set -euo pipefail

required_file="coco_captioning/val2014_vgg16_fc7_pca.h5"

if [ ! -f "$required_file" ]; then
    if [ ! -f "coco_captioning.zip" ]; then
        wget "http://cs231n.stanford.edu/coco_captioning.zip"
    fi
    unzip -o coco_captioning.zip
fi
