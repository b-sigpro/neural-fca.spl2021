# Neural FCA for Blind Source Separation
This repository provides inference scripts for Neural FCA proposed in our paper: [Neural Full-rank Spatial Covariance Analysis for Blind Source Separation](https://ieeexplore.ieee.org/document/9506855).

Please cite as:
```bibtex
@article{bando2021neural,
  title={Neural Full-Rank Spatial Covariance Analysis for Blind Source Separation},
  author={Bando, Yoshiaki and Sekiguchi, Kouhei and Masuyama, Yoshiki and Nugraha, Aditya Arie and Fontaine, Mathieu and Yoshii, Kazuyoshi},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={1670--1674},
  year={2021},
  publisher={IEEE}
}
```

## Environments
Neural FCA was developed with Python 3.8 and the following requirements:
```shell
pip install -r requirements.txt
```

## Pre-trained model
The pre-trained model used in the paper for separating speech mixtures can be downloaded from the following URL:
```shell
wget https://github.com/yoshipon/neural-fca_spl2021/releases/download/v0.0.0/model.zip
unzip model.zip
```

## Source separation
1. The pre-trained model can separate four-channel two-speech mixtures as follows:
```shell
python spl2021_neural-fca/separate.py model/ input.wav output.wav
```
The model will predict three sources assuming two target sources and one noise source.

2. If you want to perform the separation without inference-time parameter updates, run the following command:
```shell
python spl2021_neural-fca/separate.py model/ input.wav output.wav --n_iter=0
```

3. You can obtain `neural-fca.png` showing mixture and source spectrograms by:
```shell
python spl2021_neural-fca/separate.py model/ input.wav output.wav --n_iter=0 --plot
```

## License
This repository is released under the MIT License.
The pre-trained model is released under the Creative Commons BY-NC-ND 4.0 License.

## Contact
Yoshiaki Bando, y.bando@aist.go.jp

National Institute of Advanced Industrial Science and Technology, Japan
