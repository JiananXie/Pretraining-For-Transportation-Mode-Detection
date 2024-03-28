# Pre-training for transportation mode detection

created for usage of project

### Log
  - 2023.10.11 upload the original code
  - 2023.10.19 update some materials about Bert
  - 2023.10.30 basic bert completed
  - 2023.11.13 first attempt of verification of the effect of pretrained BERT on TMD
  - 2023.12.1 update stage2 code into new branch, which try to use grid input and add more metrics on model
  - 2023.12.10-12.15 first attempt to complete Methodology and Experiment
  - 2023.12.18-12.24 supplement to Experiment and first attemp to complete Introduction,  improve the performance of grid input with pseudo Mercator process
  - 2023.12.25-12.31 complete the whole work and the draft of the paper
***
接下来每天更新工作以记录进度：(from 2024.3.16)
  - 2024.3.16 读文章：`PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting， Leveraging Language Foundation Models for Human Mobility Forecasting`； 项目：复现`Dabiri et al.(2019)`工作
  - 2024.3.17 复现`Dabiri et al.(2019)`， 问题：数据预处理出错
  - 2024.3.18 数据处理问题解决(作者提供代码有误，在issues可以找到解决方法)，成功跑通论文中SECA和supervised CNN方法
  - 2024.3.19 成功跑通`Dabiri et al.(2019)`的baseline，准备着手复现`Leveraging the Self-Transition Probability of Ordinal Patterns Transition Network for Transportation Mode Identification Based on GPS Data`(2020)，同时研究Foursquare数据集
  - 2024.3.20 复现OP-TMI，研究论文，以及寻找新GPS数据集
  -2024.3.21 复现OP-TMI，研究论文，以及寻找新GPS数据集
  - 2024.3.22 复现OP-TMI，读文章：`Trajectory as a Sequence: A novel travel mode identification framework(2023)`
  - 2024.3.24 跑通OP-TMI但需要修改mode，重新运行中。尝试H3编码，因为大部分gird\_num的设置具有局限性，在跨度较大时无法合理的设置格子大小，但问题在于对于geolife而言处理后的h3编码有32000+，大于可以添加的词汇表数量。尝试方向：1.限制所有实验中geolife数据集在北京内，2.引入speed，acceleration，jerk等信息，在传入bert前先进行卷积编码。对处理后数据集进行了简单可视化，发现有很多新疆等地收集的数据，也许我们得尝试第二种方式。
  - 2024.3.25 读文章:`Semi-Supervised Deep Learning Approach for Transportation Mode Identification Using GPS Trajectory Data(2019)`、`Leveraging the Self-Transition Probability of Ordinal Patterns Transition Network for Transportation Mode Identification Based on GPS Data(2022)`。
    - **缺少change point检测，之前没有考虑过实际检测轨迹mode时，大多数情况是多mode的，项目仅简单的在预知change point(从标好的数据集中分割)情况下进行了TMD任务的学习**
  - 2024.3.26 读文章:`Trajectory as a Sequence: A novel travel mode identification framework(2023)`, `DeepStay: Stay Region Extraction from Location Trajectories using
Weak Supervision(2023)`, `An Ensemble of ConvTransformer Networks for the Sussex-Huawei Locomotion-Transportation (SHL) Recognition Challenge(2021)`
    - **1.TMI领域最新两篇利用了transformer的：`A deep learning approach for transportation mode identification using a transformation of GPS trajectory data features into an image representation(2024 preprint)`,`DeepStay: Stay Region Extraction from Location Trajectories using Weak Supervision(2023)`**
    - **2.发表的文章中目前能找到开源还没实现的有`A novel one-stage approach for pointwise transportation mode identification inspired by point cloud processing(2023)`**
    - **3.如何处理分段问题？工作聚焦于对给定segment分类？(一段取众数？自动分割不同mode？)**
    - **4.SHL和GL的处理暂时不同**
    - **5.是否可以利用半监督？如果仅聚焦于trm encoder比CAE/CE学习表征的能力更强，从而利用大量的未标记数据**
    - **6.输入只利用格网编号？H3？还是Cov-transformer？还是只利用speed？**
  - 2024.3.27 读文章：`A novel one-stage approach for pointwise transportation mode identification inspired by point cloud processing(2023)`，成功跑通`Dabiri et al.(2019)`的所有方法，重新整理了一遍格网预处理部分，发现以前的代码有问题，现改为大圆等距分割。回答：为什么一般工作都采用motion features，而不是直接用gps数据？**08年geolife作者只讲了对于raw gps data，利用机器学习方法去做检测，而第一步的feature extraction我们采用了.....等特征，并没有阐述清楚为什么不直接用gps数据**
    -  **再寻找相关领域是否有直接利用gps数据或者格网输入的先例**
  - 2024.3.28 修改后格网编号太大，目前效果很差，但反而利用speed单个特征能做到最好的表现，后面需要研究原因，词表无法利用还是speed分词后较短，能比较好学习到。
