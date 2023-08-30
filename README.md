# SAM-Med2D \[[Paper]()]
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/litianbin/SAM-Med2D)
</a> 
<a src="https://img.shields.io/badge/cs.CV-2305.06356-b31b1b?logo=arxiv&logoColor=red" href=" "> <img src="https://img.shields.io/badge/cs.CV-2305.06356-b31b1b?logo=arxiv&logoColor=red">
<a src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat" href="https://github.com/uni-medical/SAM-Med2D/blob/main/assets/SAM-Med2D_wechat_group.jpeg"> <img src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat">
</a>
[![GitHub Stars](https://img.shields.io/github/stars/uni-medical/SAM-Med2D.svg?style=social&label=Star&maxAge=60)](https://github.com/uni-medical/SAM-Med2D)🔥🔥🔥
<!-- ## Description -->

## 🌤️ Highlights
- 🏆 Collected and curated the largest medical image segmentation dataset (4.6M images and 19.7M masks) to date for training models.
- 🏆 The most comprehensive fine-tuning based on Segment Anything Model (SAM).
- 🏆 Comprehensive evaluation of SAM-Med2D on large-scale datasets.

## Dataset 👈
SAM-Med2D is trained and tested on a dataset that includes **4.6M images** and **19.7M masks**. This dataset covers 10 medical data modalities, 4 anatomical structures + lesions, and 31 major human organs. To our knowledge, this is currently the largest and most diverse medical image segmentation dataset in terms of quantity and coverage of categories.
<p align="center"><img width="800" alt="image" src="https://github.com/uni-medical/SAM-Med2D/blob/main/assets/dataset.png"></p> 

## Framwork 👈
The pipeline of SAM-Med2D. We freeze the image encoder and incorporate learnable adapter layers in each Transformer block to acquire domain-specific knowledge in the medical field. We fine-tune the prompt encoder using point, Bbox, and mask information, while updating the parameters of the mask decoder through interactive training.
<p align="center"><img width="800" alt="image" src="https://github.com/uni-medical/SAM-Med2D/blob/main/assets/framwork.png"></p> 

## Results 👈
<p align="center"><img width="800" alt="image" src="https://github.com/uni-medical/SAM-Med2D/blob/main/assets/result.png"></p> 

## 🚀 Online Demo

**SAM-Med2D** online Demo can be found on [OpenXLab](https://openxlab.org.cn/apps/detail/litianbin/SAM-Med2D). Let's try it!


## 🔥 Updates
- (2023.08.26) Online Demo release
- (2023.08.31) Paper release

## 🗓️ Ongoing
- [x] Online Demo release
- [x] Paper release
- [ ] Code release

## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE). 

## 💬 Discussion Group
If you have any questions about SAM-Med2D, feel free to join our WeChat group discussion:

<p align="center"><img width="300" alt="image" src="https://github.com/uni-medical/SAM-Med2D/blob/main/assets/SAM-Med2D_wechat_group.jpeg"></p> 

## 🤝 Acknowledgement
Thanks to the open-source of the following projects:   
[Segment Anything](https://github.com/facebookresearch/segment-anything) &#8194;

## 👋 Hiring
We are hiring researchers, engineers and interns in **General Vision Group, Shanghai AI Lab**. If you are interested in Medical Foundation Models and General Medical AI, including designing benchmark datasets , general models, evaluation systems, and efficient tools, please contact [Junjun He](https://scholar.google.com/citations?hl=zh-CN&user=Z4LgebkAAAAJ)(`hejunjun@pjlab.org.cn`), [Jin Ye](https://scholar.google.com/citations?user=UFBrJOAAAAAJ&hl=zh-CN)(`yejin@pjlab.org.cn`), and Tianbin Li (`litianbin@pjlab.org.cn`).
