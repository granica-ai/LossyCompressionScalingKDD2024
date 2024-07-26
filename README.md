# Scaling Training Data with Lossy Image Compression 
*Katherine L. Mentzer and Andrea Montanari*  

Accepted to KDD '24  

[arXiv Preprint](https://arxiv.org/abs/2407.17954)  
[Promo Video](https://www.youtube.com/watch?v=lnfAwDadx-8)

## Summary 
Scaling Training Data with Lossy Image Compression addresses how to handle large computer vision datasets when storage is limited. Storing massive datasets can be costly and challenging, but there are options to optimize storage with lossy image compression. We explore the balance between keeping fewer, full-resolution images and keeping more, lossily-compressed images. Our findings reveal that lossy compression can improve performance in the storage-limited regime. Understanding the relationship between test error, number of images, and compression levels allows us to optimize storage and improve model performance. This approach is valuable for anyone facing storage limitations. Learn more in the full paper.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/repo-name.git
   cd repo-name
   ```

2. Create the main environment:
    ```bash
    conda env create -f environment.yaml
    ```

3. Create the MMSegmentation and MMDetection environment:
    ```bash
    conda env create -f environment_mm.yaml
    ```
    See the official [MMSegmentation Installation Instructions](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) and [MMDetection Installation Instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html) for more information on how to install the necessary packages. 

## Usage

The scripts are divided into classification, segmentation, detection, and theory for each set of experiments. Scripts with the prefix `train` are used for model training. The `paper_plots.ipynb` file shows the creation of each of the figures in the paper. 

## Datasets
- [Food101 Dataset](https://huggingface.co/datasets/ethz/food101)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [iSAID Dataset](https://captain-whu.github.io/iSAID/)

## Contact
For any questions or inquiries, please contact Katherine Mentzer at [kaleigh.mentzer@granica.ai](mailto:kaleigh.mentzer@granica.ai).