# Semantic Segmentation on Carvana Dataset

In this project, I use an adaptation of U-Net with residual connections. This particular implementation of U-Net is lighter as well. 

![Image with Predictions!](/imgs/doc_predictions.png "Image with Predictions")


## Network Architecture

I utilized the general structure of the network with the same number of downsamples with 3 skip-connections. 
U-Net architecture:  [paper link](https://arxiv.org/pdf/1505.04597.pdf)

![Image with Predictions!](/imgs/doc_unet.png "Image with Predictions")

Each block additionally contains a residual connection as well to pass previous information along throughout the network for improvements. The general procedure for a residual connection can be found here:

![Residual connections!](/imgs/doc_resconnection.png "Residual connections")