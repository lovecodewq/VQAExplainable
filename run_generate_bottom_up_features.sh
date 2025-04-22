#! /bin/bash

python generate_bottom_up_features.py \
    --output-dir ./bottom_up_features/resnet \
    --feature_type resnet \
    --device cpu \
    --split minitrain
