#! /bin/bash
python -m train -gpu -logdir training_logs/ -rt --save-every 2 \
--model unet --epochs 4 --batch-size 64 -lr 1e-3 --optimizer Adam \
--scheduler Reduce --workers 2 --resize 256