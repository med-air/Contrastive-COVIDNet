# Multi-site COVID-Net CT Classification
This is the PyTorch implemention of our paper [**Contrastive Cross-site Learning with Redesigned Net for COVID-19 CT Classification**](https://ieeexplore.ieee.org/document/9194240) by [Zhao Wang](http://kyfafyd.wang/), [Quande Liu](https://liuquande.github.io/), [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou/)

## Abatract

> This paper proposes a novel joint learning framework to perform accurate COVID-19 identification by effectively learning with heterogeneous datasets with distribution descrepancy. We build a powerful backbone by redesigning the recently proposed COVID-Net in aspects of network architecture and learning strategy to improve the prediction accuracy and learning efficiency. On top of it, we further explicitly tackle the cross-site domain shift by conducting separate feature normalization in latent space. Moreover, we propose a con-trastive training objective to enhance the domain invariance of semantic embeddings for boosting the classification performance on each dataset. We develop and evaluate our method with two public large-scale COVID-19 diagnosis datasets from real CT images. Extensive experiments show that our approach consistently improves the performances on both datasets, as well as outperforms existing state-of-the-art multi-site learning methods.

![avatar](assets/framework.png)

## Usage

#### Setup

We suggest using Anaconda to setup environment on Linux, if you have installed anaconda, you can skip this step.
```shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh && zsh Anaconda3-2020.11-Linux-x86_64.sh
```
Then, we can install packages using provided `environment.yaml`.
```shell
git clone https://github.com/med-air/Contrastive-COVIDNet
cd Contrastive-COVIDNet
conda env create -f environment.yaml
conda activate pytorch0.4.1
```

#### Dataset

We employ two publicly available COVID-19 CT datasets:

- [SARS-CoV-2 dataset](https://www.medrxiv.org/content/10.1101/2020.04.24.20078584v3)
- [COVID-CT dataset](http://arxiv.org/abs/2003.13865)

Download our pre-processed datasets from [Google Drive](https://drive.google.com/file/d/1JBp9RH9-yBEdtkNYDi6wWL79o62JD5Td/view?usp=sharing) and put into `data/` directory.

#### Pretrained Model

You can directly download our pretrained model from [Google Drive](https://drive.google.com/file/d/1ZwtxF4c_pvyv_uyE4Zx4_bNNHQx7Y_Ao/view?usp=sharing) and put into `saved/` directory for testing.

#### Training

```shell
cd code
python main.py --bna True --bnd True --cosine True --cont True
```

#### Test

```shell
cd code
python test.py
```

## Citation
If you find this code and dataset useful, please cite in your research papers.
```
@article{wangcontrastive,
   author={Wang, Zhao and Liu, Quande and Dou, Qi},
   title={Contrastive Cross-site Learning with Redesigned Net for COVID-19 CT Classification},
   journal={IEEE Journal of Biomedical and Health Informatics},
   DOI={10.1109/jbhi.2020.3023246},
   year={2020},
   volume={24},
   number={10},
   pages={2806-2813}
}
```


## Questions

For further questions, pls feel free to contact [Zhao Wang](mailto:kyfafyd@zju.edu.cn)

## References

[1] E. Soares, P. Angelov, S. Biaso, M. Higa Froes, and D. Kanda Abe, “Sars-cov-2 ct-scan dataset: A large dataset of real patients ct scans for sars-cov-2 identification,” medRxiv, 2020.

[2] J. Zhao, X. He, X. Yang, Y. Zhang, S. Zhang, and P. Xie, “Covid-ct-dataset: A ct scan dataset about covid-19,” 2020.

[3] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga *et al.*, “Pytorch: An imperative style, high-performance deep learning library,” in *Advances in neural information processing systems*, 2019, pp. 8026–8037.
