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
  - 2024.3.16 读文章：PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting， Leveraging Language Foundation Models for Human Mobility Forecasting； 项目：复现Dabiri et al.(2019)工作
  - 2024.3.17 复现Dabiri et al.(2019)， 问题：数据预处理出错
  - 2024.3.18 数据处理问题解决(作者提供代码有误，在issues可以找到解决方法)，成功跑通论文中SECA和supervised CNN方法
  - 2024.3.19 成功跑通Dabiri et al.(2019)的baseline，准备着手复现`Leveraging the Self-Transition Probability of Ordinal Patterns Transition Network for Transportation Mode Identification Based on GPS Data`(2020)，同时研究Foursquare数据集
