# SAM-Med2D \[[Paper](https://arxiv.org/abs/2308.16184)]
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/GMAI/SAM-Med2D)
</a> 
<a src="https://img.shields.io/badge/Data-SAMed2D_20M-blue?logo=red" href="https://openxlab.org.cn/datasets/GMAI/SA-Med2D-20M"> <img src="https://img.shields.io/badge/Data-SAMed2D_20M-blue?logo=red">
<a src="https://img.shields.io/badge/cs.CV-2308.16184-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2308.16184"> <img src="https://img.shields.io/badge/cs.CV-2308.16184-b31b1b?logo=arxiv&logoColor=red">
<a src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat" href="https://github.com/OpenGVLab/SAM-Med2D/blob/main/assets/SAM-Med2D_wechat_group.jpeg"> <img src="https://img.shields.io/badge/WeChat-Group-green?logo=wechat">
</a>
<a target="_blank" href="https://colab.research.google.com/github/OpenGVLab/SAM-Med2D/blob/main/predictor_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
[![GitHub Stars](https://img.shields.io/github/stars/OpenGVLab/SAM-Med2D.svg?style=social&label=Star&maxAge=60)](https://github.com/OpenGVLab/SAM-Med2D)🔥🔥🔥
<!-- ## Description -->

## 🌤️ Highlights
- 🏆 Collected and curated the largest medical image segmentation dataset (4.6M images and 19.7M masks) to date for training models.
- 🏆 The most comprehensive fine-tuning based on Segment Anything Model (SAM).
- 🏆 Comprehensive evaluation of SAM-Med2D on large-scale datasets.

## 🔥 Updates
- (2023.12.05) We open the download of the dataset on the [Hugging Face](https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M) platform 
- (2023.11.23) We have released the [SA-Med2D-20M](https://openxlab.org.cn/datasets/GMAI/SA-Med2D-20M) dataset 
- (2023.11.21) We have released article introducing the [SA-Med2D-20M](https://arxiv.org/abs/2311.11969) dataset 
- (2023.10.24) We now released [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D), which focus on segmentation of 3D medical imaging
- (2023.09.14) Train code release
- (2023.09.02) Test code release
- (2023.08.31) Pre-trained model release
- (2023.08.31) Paper release
- (2023.08.26) Online Demo release

## 👉 Dataset
SAM-Med2D is trained and tested on a dataset that includes **4.6M images** and **19.7M masks**. This dataset covers 10 medical data modalities, 4 anatomical structures + lesions, and 31 major human organs. To our knowledge, this is currently the largest and most diverse medical image segmentation dataset in terms of quantity and coverage of categories.
<p align="center"><img width="800" alt="image" src="https://github.com/OpenGVLab/SAM-Med2D/blob/main/assets/dataset.png"></p> 

## 👉 Framework
The pipeline of SAM-Med2D. We freeze the image encoder and incorporate learnable adapter layers in each Transformer block to acquire domain-specific knowledge in the medical field. We fine-tune the prompt encoder using point, Bbox, and mask information, while updating the parameters of the mask decoder through interactive training.
<p align="center"><img width="800" alt="image" src="https://github.com/OpenGVLab/SAM-Med2D/blob/main/assets/framwork.png"></p> 

## 👉 Results

<table>
  <caption align="center">Quantitative comparison of different methods on the test set: </caption>
  <thead>
    <tr>
      <th>Model</th>
      <th>Resolution</th>
      <th>Bbox (%)</th>
      <th>1 pt (%)</th>
      <th>3 pts (%)</th>
      <th>5 pts (%)</th>
      <th>FPS</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">SAM</td>
      <td align="center">$256\times256$</td>
      <td align="center">61.63</td>
      <td align="center">18.94</td>
      <td align="center">28.28</td>
      <td align="center">37.47</td>
      <td align="center">51</td>
      <td align="center"><a href="https://drive.google.com/file/d/1_U26MIJhWnWVwmI5JkGg2cd2J6MvkqU-/view?usp=drive_link">Offical</a></td>
    </tr>
    <tr>
      <td align="center">SAM</td>
      <td align="center">$1024\times1024$</td>
      <td align="center">74.49</td>
      <td align="center">36.88</td>
      <td align="center">42.00</td>
      <td align="center">47.57</td>
      <td align="center">8</td>
      <td align="center"><a href="https://drive.google.com/file/d/1_U26MIJhWnWVwmI5JkGg2cd2J6MvkqU-/view?usp=drive_link">Offical</a></td>
    </tr>
    <tr>
      <td align="center">FT-SAM</td>
      <td align="center">$256\times256$</td>
      <td align="center">73.56</td>
      <td align="center">60.11</td>
      <td align="center">70.95</td>
      <td align="center">75.51</td>
      <td align="center">51</td>
      <td align="center"><a href="https://drive.google.com/file/d/1J4qQt9MZZYdv1eoxMTJ4FL8Fz65iUFM8/view?usp=drive_link">FT-SAM</a></td>
    </tr>
    <tr>
      <td align="center">SAM-Med2D</td>
      <td align="center">$256\times256$</td>
      <td align="center">79.30</td>
      <td align="center">70.01</td>
      <td align="center">76.35</td>
      <td align="center">78.68</td>
      <td align="center">35</td>
      <td align="center"><a href="https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link">SAM-Med2D</a></td>
    </tr>
  </tbody>
</table>

百度云链接: https://pan.baidu.com/s/1HWo_s8O7r4iQI6irMYU8vQ?pwd=dk5x
提取码: dk5x 


<table>
    <caption align="center">Generalization validation on 9 MICCAI2023 datasets, where "*" denotes that we drop adapter layer of SAM-Med2D in test phase: </caption>
  <thead>
    <tr>
      <th rowspan="2">Datasets</th>
      <th colspan="3">Bbox prompt (%)</th>
      <th colspan="3">1 point prompt (%)</th>
    </tr>
    <tr>
      <th>SAM</th>
      <th>SAM-Med2D*</th>
      <th>SAM-Med2D</th>
      <th>SAM</th>
      <th>SAM-Med2D*</th>
      <th>SAM-Med2D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><a href="https://www.synapse.org/#!Synapse:syn51236108/wiki/621615">CrossMoDA23</a></td>
      <td align="center">78.12</td>
      <td align="center">86.26</td>
      <td align="center">88.42</td>
      <td align="center">33.84</td>
      <td align="center">65.85</td>
      <td align="center">85.26</td>
    </tr>
    <tr>
      <td align="center"><a href="https://kits-challenge.org/kits23/">KiTS23</a></td>
      <td align="center">81.52</td>
      <td align="center">86.14</td>
      <td align="center">89.89</td>
      <td align="center">31.36</td>
      <td align="center">56.67</td>
      <td align="center">83.71</td>
    </tr>
    <tr>
      <td align="center"><a href="https://codalab.lisn.upsaclay.fr/competitions/12239#learn_the_details">FLARE23</a></td>
      <td align="center">73.20</td>
      <td align="center">77.18</td>
      <td align="center">85.09</td>
      <td align="center">19.87</td>
      <td align="center">32.01</td>
      <td align="center">77.17</td>
    </tr>
    <tr>
      <td align="center"><a href="https://atlas-challenge.u-bourgogne.fr/">ATLAS2023</a></td>
      <td align="center">76.98</td>
      <td align="center">79.09</td>
      <td align="center">82.59/td>
      <td align="center">29.07</td>
      <td align="center">45.25</td>
      <td align="center">64.76</td>
    </tr>
    <tr>
      <td align="center"><a href="https://multicenteraorta.grand-challenge.org/">SEG2023</a></td>
      <td align="center">64.82</td>
      <td align="center">81.85</td>
      <td align="center">85.09</td>
      <td align="center">21.15</td>
      <td align="center">34.71</td>
      <td align="center">72.08</td>
    </tr>
    <tr>
      <td align="center"><a href="https://lnq2023.grand-challenge.org/lnq2023/">LNQ2023</a></td>
      <td align="center">53.02</td>
      <td align="center">57.37</td>
      <td align="center">58.01</td>
      <td align="center">7.05</td>
      <td align="center">7.21</td>
      <td align="center">37.64</td>
    </tr>
    <tr>
      <td align="center"><a href="https://codalab.lisn.upsaclay.fr/competitions/9804">CAS2023</a></td>
      <td align="center">61.53</td>
      <td align="center">78.20</td>
      <td align="center">81.10</td>
      <td align="center">22.75</td>
      <td align="center">46.85</td>
      <td align="center">78.46</td>
    </tr>
    <tr>
      <td align="center"><a href="https://tdsc-abus2023.grand-challenge.org/Dataset/">TDSC-ABUS2023</a></td>
      <td align="center">64.31</td>
      <td align="center">69.00</td>
      <td align="center">66.14</td>
      <td align="center">8.24</td>
      <td align="center">18.98</td>
      <td align="center">43.55</td>
    </tr>
    <tr>
      <td align="center"><a href="https://toothfairy.grand-challenge.org/toothfairy/">ToothFairy2023</a></td>
      <td align="center">43.40</td>
      <td align="center">39.13</td>
      <td align="center">41.23</td>
      <td align="center">5.47</td>
      <td align="center">5.27</td>
      <td align="center">12.93</td>
    </tr>
    <tr>
      <td align="center">Weighted sum</td>
      <td align="center">73.49</td>
      <td align="center">77.67</td>
      <td align="center">84.88</td>
      <td align="center">20.88</td>
      <td align="center">34.30</td>
      <td align="center">76.63</td>
    </tr>
  </tbody>
</table>

##  Typos of paper
In our original paper, we acknowledge that there were anomalies in the test data presented in Table 4. We have conducted data updates for this project and corrected the values in Table 4. We assure the readers that our research team has recognized this issue and will update Table 4 in the next version. We apologize for any inconvenience this may have caused.

## 👉 Visualization
<p align="center"><img width="800" alt="image" src="https://github.com/OpenGVLab/SAM-Med2D/blob/main/assets/visualization.png"></p> 

## 👉 Train
Prepare your own dataset and refer to the samples in `SAM-Med2D/data_demo` to replace them according to your specific scenario. You need to generate the `image2label_train.json` file before running `train.py`.

If you want to use mixed-precision training, please install [Apex](https://github.com/NVIDIA/apex). If you don't want to install Apex, you can comment out the line `from apex import amp` and set `use_amp` to False.

```bash
cd ./SAM-Med2D
python train.py
```
- work_dir: Specifies the working directory for the training process. Default value is `workdir`.
- image_size: Default value is 256.
- mask_num: Specify the number of masks corresponding to one image, with a default value of 5.
- data_path: Dataset directory, for example: `data_demo`.
- resume: Pretrained weight file, ignore `sam_checkpoint` if present.
- sam_checkpoint: Load sam checkpoint.
- iter_point: Mask decoder iterative runs.
- multimask: Determines whether to output multiple masks. Default value is True.
- encoder_adapter: Whether to fine-tune the Adapter layer, set to False only for fine-tuning the decoder.
- use_amp: Set whether to use mixed-precision training.

## 👉 Test
Prepare your own dataset and refer to the samples in `SAM-Med2D/data_demo` to replace them according to your specific scenario. You need to generate the `label2image_test.json` file before running `test.py`.

```bash
cd ./SAM-Med2D
python test.py
```
- work_dir: Specifies the working directory for the testing process. Default value is `workdir`.
- batch_size: 1.
- image_size: Default value is 256.
- boxes_prompt: Use Bbox prompt to get segmentation results. 
- point_num: Specifies the number of points. Default value is 1.
- iter_point: Specifies the number of iterations for point prompts.
- sam_checkpoint: Load sam or sammed checkpoint.
- encoder_adapter: Set to True if using SAM-Med2D's pretrained weights.
- save_pred: Whether to save the prediction results.
- prompt_path: Is there a fixed Prompt file? If not, the value is None, and it will be automatically generated in the latest prediction.

## 👉 Deploy

### Export to ONNX

- export encoder model

```bash
python3 scripts/export_onnx_encoder_model.py --sam_checkpoint /path/to/sam-med2d_b.pth --output /path/to/sam-med2d_b.encoder.onnx --model-type vit_b --image_size 256 --encoder_adapter True
```

- export decoder model

```bash
python3 scripts/export_onnx_model.py --checkpoint /path/to/sam-med2d_b.pth --output /path/to/sam-med2d_b.decoder.onnx --model-type vit_b --return-single-mask
```

- inference with onnxruntime

```
# cd examples/SAM-Med2D-onnxruntime
python3 main.py --encoder_model /path/to/sam-med2d_b.encoder.onnx --decoder_model /path/to/sam-med2d_b.decoder.onnx
```

## 🚀 Try SAM-Med2D
- 🏆 **Gradio Online:** Online Demo can be found on [OpenXLab](https://openxlab.org.cn/apps/detail/GMAI/SAM-Med2D).
- 🏆 **Notebook Demo:** You can use [predictor_example.ipynb](https://github.com/OpenGVLab/SAM-Med2D/blob/main/predictor_example.ipynb) to run it locally to view the prediction results generated by different prompts.
- 🏆 **Gradio Local:** You can deploy [app.ipynb](https://github.com/OpenGVLab/SAM-Med2D/blob/main/app.ipynb) locally and upload test cases.
- **Notes:** Welcome to feedback [good case👍](https://github.com/OpenGVLab/SAM-Med2D/issues/2) and [bad case👎](https://github.com/OpenGVLab/SAM-Med2D/issues/1) in issue.

## 🗓️ Ongoing
- [x] Dataset release
- [x] Dataset article release
- [x] Train code release
- [x] Test code release
- [x] Pre-trained model release
- [x] Paper release
- [x] Online Demo release

## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE). 

## 💬 Discussion Group
If you have any questions about SAM-Med2D, please add this WeChat ID to the WeChat group discussion:

<p align="center"><img width="300" alt="image" src="https://github.com/OpenGVLab/SAM-Med2D/blob/main/assets/SAM_Med2D_wechat_group.png"></p> 

## 🤝 Acknowledgement
- We thank all medical workers and dataset owners for making public datasets available to the community.
- Thanks to the open-source of the following projects: [Segment Anything](https://github.com/facebookresearch/segment-anything) &#8194;

## 👋 Hiring & Global Collaboration
- **Hiring:** We are hiring researchers, engineers, and interns in General Vision Group, Shanghai AI Lab. If you are interested in Medical Foundation Models and General Medical AI, including designing benchmark datasets, general models, evaluation systems, and efficient tools, please contact us.
- **Global Collaboration:** We're on a mission to redefine medical research, aiming for a more universally adaptable model. Our passionate team is delving into foundational healthcare models, promoting the development of the medical community. Collaborate with us to increase competitiveness, reduce risk, and expand markets.
- **Contact:** Junjun He(hejunjun@pjlab.org.cn), Jin Ye(yejin@pjlab.org.cn), and Tianbin Li (litianbin@pjlab.org.cn).

## Reference
```
@misc{cheng2023sammed2d,
      title={SAM-Med2D}, 
      author={Junlong Cheng and Jin Ye and Zhongying Deng and Jianpin Chen and Tianbin Li and Haoyu Wang and Yanzhou Su and
              Ziyan Huang and Jilong Chen and Lei Jiangand Hui Sun and Junjun He and Shaoting Zhang and Min Zhu and Yu Qiao},
      year={2023},
      eprint={2308.16184},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{ye2023samed2d20m,
      title={SA-Med2D-20M Dataset: Segment Anything in 2D Medical Imaging with 20 Million masks}, 
      author={Jin Ye and Junlong Cheng and Jianpin Chen and Zhongying Deng and Tianbin Li and Haoyu Wang and Yanzhou Su and Ziyan Huang and Jilong Chen and Lei Jiang and Hui Sun and Min Zhu and Shaoting Zhang and Junjun He and Yu Qiao},
      year={2023},
      eprint={2311.11969},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
