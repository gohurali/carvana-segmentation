# Semantic Segmentation on Carvana Dataset

In this project, I use an adaptation of U-Net with residual connections. This particular implementation of U-Net is lighter as well. 

![Image with Predictions!](/imgs/doc_predictions.png "Image with Predictions")


## Network Architecture

I utilized the general structure of the network with the same number of downsamples with 3 skip-connections. Both of my implementations will down sample by a factor of 32.

![unet arch!](/imgs/doc_unet.png "unet arch")

Each block additionally contains a residual connection as well to pass previous information along throughout the network for improvements. The general procedure for a residual connection can be found here:

![Residual connections!](/imgs/doc_resconnection.png "Residual connections")

## Training

I primilarily utilized Google Colab for training thus was restricted to utilizing a NVIDIA Tesla K80 GPU.

Run the training with the following:
```
python -m train -gpu -logdir training_logs/ --save-every 2 \
--model unet --epochs 10 --batch-size 64 -lr 1e-3 --optimizer Adam \
--scheduler Reduce --workers 2 --resize 256
```

## Referenced Papers:

Many ideas were formulated from the following papers:

[Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)  
[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385v1.pdf)  
[U-Net: Convolutional Networks for Biomedical
Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)  


