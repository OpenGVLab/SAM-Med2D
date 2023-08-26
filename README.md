# SAM-Med2D [Paper (Coming soon)]
<!-- ## Description -->
We introduce **SAM-Med2D**, the first SAM fully fine-tuned on a large-scale medical dataset. Specifically, we first collect and curate approximately 4.6M images and 19.7M masks from 215 publicly available datasets, constructing a large-scale medical image segmentation dataset encompassing various patterns and objects.
Then, we train the SAM-Med2D model on this dataset using an interactive segmentation approach. Unlike previous methods that only provide bounding box or point prompts, we adapt SAM to medical image segmentation through a more comprehensive prompt involving bounding boxes, points, and masks.
Finally, we conducted a comprehensive evaluation and analysis to investigate the performance of SAM-Med2D in medical image segmentation across various modalities, anatomical structures, and organs. Concurrently, we validated the generalization capability of SAM-Med2D on nine datasets from MICCAI 2023 challenge. Overall, our approach demonstrated significantly superior performance and generalization capability compared to SAM and other fine-tuning methods. 


## 🚀 Online Demo
**SAM-Med2D** online Demo can be found on [OpenXLab](https://openxlab.org.cn/apps/detail/litianbin/SAM-Med2D). Let's try it!


## 🔥 Updates
- (2023.08.26) Online Demo release


## 🗓️ Ongoing
- [ ] Paper & Code release
- [x] Online Demo release


## 🎫 License

This project is released under the [Apache 2.0 license](LICENSE). 

## 💬 Discussion Group

If you have any questions about SAM-Med2D, feel free to join our WeChat group discussion:

<p align="center"><img width="300" alt="image" src="https://github.com/uni-medical/SAM-Med2D/blob/main/assets/SAM-Med2D_wechat_group.jpeg"></p> 

## 🤝 Acknowledgement
Thanks to the open-source of the following projects:   

[Segment Anything](https://github.com/facebookresearch/segment-anything) &#8194;
