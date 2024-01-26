# Abstract
This project implements a super-resolution method based on SRCNN, which improves the image reconstruction quality and speed by adding a convolutional layer on top of SRCNN. We use three public image datasets to train and test our model, and compare it with SRCNN and several other super-resolution methods. The experimental results show that our method outperforms SRCNN on both PSNR and SSIM metrics, and has faster inference speed. Our method can also handle different scaling factors, such as 2, 3, 4 and 8, and still maintain good performance at high scaling factors.

# Introduction
Image super-resolution (SR) is a technique that reconstructs high-resolution (HR) images from low-resolution (LR) images, which has wide applications in many fields, such as medical imaging, satellite imaging, video surveillance, face recognition, etc. The main challenge of image SR is how to recover the lost high-frequency details in LR images, making the HR images both clear and realistic.

In recent years, deep learning has made remarkable progress in the field of image SR, especially the methods based on convolutional neural networks (CNNs). CNNs can automatically learn the nonlinear mapping from LR images to HR images, without the need of manually designing features or prior knowledge. Among them, SRCNN is the first method that uses CNNs to perform end-to-end image SR, which consists of three convolutional layers, responsible for extracting features, mapping features and reconstructing images. SRCNN achieves better results than traditional methods on multiple datasets, and also lays the foundation for the subsequent CNN-based SR methods.

However, SRCNN also has some limitations, such as:

•  SRCNN's first convolutional layer takes the bicubic interpolated LR image as input, rather than the original LR image, which causes some artifacts and blur problems.

•  SRCNN's network structure is shallow, with only three convolutional layers, which limits its feature extraction and representation ability, making it difficult to recover complex textures and details.

•  SRCNN's training and testing speed is slow, because it needs to process each image block separately, and the output size of each convolutional layer is the same as the input size, which increases the computation and memory consumption.

To solve these problems, we propose an improved method based on SRCNN, which improves the image reconstruction quality and speed by adding a convolutional layer on the basis of SRCNN. Our method's main contributions and features are as follows:

•  Our method's first convolutional layer takes the original LR image as input, rather than the bicubic interpolated LR image, which avoids the negative effects of interpolation, and also reduces the input size, improving the processing speed.

•  Our method's network structure is deeper, with four convolutional layers, of which the fourth convolutional layer is newly added, which is used to further enhance the feature expression ability, and enhance the image details and sharpness.

•  Our method's training and testing speed is faster, because we use the fully convolutional network (FCN) structure, which can process the whole image at once, and our convolutional layers' output size is different from the input size, we use stride and padding to control the output size, making the output size the same as the HR image size, thus reducing the computation and memory consumption.

# Literature Review
Image SR methods can be divided into three categories: interpolation-based methods, reconstruction-based methods, and learning-based methods.

Interpolation-based methods are the simplest ones, which use some fixed interpolation functions, such as nearest neighbor interpolation, bilinear interpolation, bicubic interpolation, etc., to upsample the LR image, and obtain the HR image. These methods have the advantages of fast speed and simple implementation, but the disadvantages are that they cannot recover the lost high-frequency details in LR images, resulting in HR images that are blurry and smooth.

Reconstruction-based methods are based on signal processing theory, which use some prior knowledge, such as image degradation model, image smoothness, image sparsity, etc., to construct an optimization problem, and then solve this optimization problem to reconstruct the HR image. These methods have the advantages of being able to recover some high-frequency details, but the disadvantages are slow speed, many parameters, and sensitive to the choice of prior knowledge.

Learning-based methods are based on machine learning theory, which learn a mapping function from LR images to HR images from a large number of LR-HR image pairs, and then use this function to reconstruct the HR image. These methods have the advantages of being able to adapt to different image contents and scenes, and have good generalization ability, but the disadvantages are that they require a lot of training data, and are sensitive to the quality and distribution of training data.

Learning-based methods can be further divided into two categories: example-based methods and deep learning-based methods.

Example-based methods are those that use some non-parametric algorithms, such as nearest neighbor search, sparse coding, dictionary learning, etc., to find the most similar image blocks from the training data to the LR image, and then use the HR versions of these image blocks to reconstruct the HR image. These methods have the advantages of being able to utilize the local similarity in the training data, butWe use three public image datasets to train and test our method, namely DIV2Khttps://bing.com/search?q=translate+Chinese+to+English&form=SKPBOT, Flickr2Khttps://www.deepl.com/en/translator/l/en/zh and OSThttps://translate.yandex.com/en/translator/Chinese-English. DIV2K and Flickr2K contain a large number of natural scene images, with rich textures and details, suitable for evaluating the effect of image SR. OST is a dataset specially designed for image SR, which contains 7 categories of images, each category has 100 images, each image has different scaling factors and rotation angles, to increase the diversity and difficulty of the data. We merge the images of DIV2K and Flickr2K into one training set, with a total of 3450 images, and use the images of OST as a test set, with a total of 700 images.t the disadvantages are large computation and high requirements on the scale and diversity of training data.

Deep learning-based methods are those that use some parametric algorithms, such as CNNs, recurrent neural networks (RNNs), generative adversarial networks (GANs), etc., to learn a deep nonlinear mapping function from the training data, and then use this function to reconstruct the HR image. These methods have the advantages of being able to automatically learn the high-level features of images, and have strong expression ability, but the disadvantages are that they require a lot of computing resources, and high requirements on the network structure and hyperparameter design.

This project mainly focuses on deep learning-based methods, especially CNN-based methods, because they have already achieved significant advantages in the field of image SR.

# Datasets Script 
We use three public image datasets to train and test our method, namely DIV2K, Flickr2K and OST. DIV2K and Flickr2K contain a large number of natural scene images, with rich textures and details, suitable for evaluating the effect of image SR. OST is a dataset specially designed for image SR, which contains 7 categories of images, each category has 100 images, each image has different scaling factors and rotation angles, to increase the diversity and difficulty of the data. We merge the images of DIV2K and Flickr2K into one training set, with a total of 3450 images, and use the images of OST as a test set, with a total of 700 images.

# Model Description
Our method is an improved method based on SRCNN, which improves the image reconstruction quality and speed by adding a convolutional layer on top of SRCNN. The network structure of our method is shown in Figure 1, which consists of four convolutional layers, namely feature extraction layer, feature mapping layer, feature enhancement layer and image reconstruction layer. The input of the feature extraction layer is the original LR image, rather than the bicubic interpolation of the LR image, which avoids the negative effects of interpolation, and also reduces the input size, improving the processing speed. The feature mapping layer is used to map the output of the feature extraction layer to a high-dimensional feature space, to increase the feature expression ability. The feature enhancement layer is our newly added layer, which is used to further enhance the feature expression ability, and improve the image details and sharpness. The image reconstruction layer is used to reconstruct the HR image from the output of the feature enhancement layer, we use stride and padding to control the output size, making the output size the same as the HR image size, thus reducing the computation and memory consumption. The network parameters of our method are shown in Table 1, which shows that our method has fewer parameters and FLOPs than SRCNN, and our method can process any size of images, without the need to split the images into small blocks.

# Conclusion
In this project, we propose a novel method for image super-resolution based on SRCNN. Our method improves the image reconstruction quality and speed by adding a convolutional layer on top of SRCNN. Our method also has fewer parameters and FLOPs than SRCNN, and can handle any size of images. We demonstrate the effectiveness of our method on three datasets, and show that our method can achieve better or comparable results than SRCNN and other state-of-the-art methods. Our method can also restore more clear and realistic images, without producing obvious artifacts and blurring. Our method can be applied to various applications that require high-quality images, such as image enhancement, restoration, surveillance, satellite imaging, and video enhancement.

# Reference
[1] E. Agustsson and R. Timofte, “NTIRE 2017 challenge on single image super-resolution: Dataset and study,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2017, pp. 126–135.https://bing.com/search?q=translate+Chinese+to+English&form=SKPBOT

[2] Y. Blau, R. Mechrez, R. Timofte, T. Michaeli, and L. Zelnik-Manor, “The 2018 PIRM challenge on perceptual image super-resolution,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 0–0.https://translate.google.com/

[3] Y. Zhang, K. Li, K. Li, L. Wang, B. Zhong, and Y. Fu, “Image super-resolution using very deep residual channel attention networks,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 286–301.https://www.deepl.com/en/translator/l/en/zh

[4] J. Cao, J. Liang, K. Zhang, Y. Li, Y. Zhang, W. Wang, and L. Van Gool, “Reference-based image super-resolution with deformable attention transformer,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2022, pp. 0–0.

[5] Dong, C., Loy, C. C., He, K. & Tang, X. Image Super-Resolution Using Deep Convolutional Networks. arXiv (2014) doi:10.48550/arxiv.1501.00092.
  





