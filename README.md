# SRCNN
Implementation of SRCNN in PyTorch.

# Usage

To train the model with a zoom factor of 2, for 200 epochs and on GPU:

`python main.py --zoom_factor 2 --nb_epoch 200 --cuda`

At each epoch, a `.pth` model file will be saved.

To use the model on an image: (the zoom factor must be the same the one used to train the model)

`python run.py --zoom_factor 2 --model model_199.pth --image example.jpg --cuda`

# Reference

[Original paper on SRCNN by Dong et al. (*Image Super-Resolution Using Deep Convolutional Networks*)](https://arxiv.org/abs/1501.00092)
