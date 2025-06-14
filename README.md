# FCHSD
中文网络仇恨言论的泛滥严重破坏了社会和谐并威胁着个体心理健康。然而，传统的粗粒度分类方法无法揭示仇恨言论的内部构成，导致其无法有效检测仇恨言论的强度和方向。针对该问题，本项目使用了一种基于大语言模型微调与提示工程相结合的细粒度抽取方案。具体而言，项目设计了包含严格格式要求的Prompt，以引导模型进行结构化输出。随后，采用LoRA技术对中文大语言模型ChatGLM-6B进行高效参数微调，使其能够学习从原始社交媒体文本到目标四元组的复杂映射关系，并能有效处理单文本内含多个仇恨实例的抽取任务。结果表明，在对测试集进行推理并与真值进行软、硬两种匹配方式的综合评估后，该方法的平均F1分数达到了0.2515。
## 数据集来源
本项目采用的数据集为STATE TOXICN，该数据集归属为大连理工大学信息检索研究室（https://ir.dlut.edu.cn/），数据集的具体构建细节详见：https://arxiv.org/abs/2501.15451。
## 环境准备
```
conda create -n FCHSD python==3.8
conda activate FCHSD
pip install transformers peft accelerate datasets torch==2.0.0
```
## 开始训练
你可以在终端中执行以下代码：
```
python run.py
```
## 模型评估
模型输出为包含2000行预测结果的txt文件（demo.txt），你可以报名参加阿里云天池大赛“CCL25-Eval 任务10：细粒度中文仇恨识别评测”（https://tianchi.aliyun.com/competition/entrance/532298/introduction）并提交demo.txt进行自动硬、软匹配平均F1分数计算。
