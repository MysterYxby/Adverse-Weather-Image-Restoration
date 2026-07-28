# Adverse-Weather-Image-Restoration
Adverse Weather Image Restoration
> 研究方向：Adverse Weather Image Restoration / Adverse Weather Image Processing  
> 适用任务：去雾、去雨、去雪、去雨滴、低照度增强、多天气统一恢复、真实退化鲁棒恢复、下游任务友好恢复等。

## 1. 研究背景

不利天气图像恢复旨在从雾、雨、雪、雨滴、夜间低照度、沙尘、霾等复杂天气条件下采集的图像中恢复更清晰、更自然、更有利于视觉理解的图像。该方向既是底层视觉任务，也是自动驾驶、智能监控、遥感、机器人感知、目标检测、语义分割和多模态感知等高层任务的重要前处理或联合优化模块。

不利天气会导致图像出现对比度下降、颜色偏移、纹理模糊、局部遮挡、反射干扰、噪声增强和结构细节丢失等问题。不同天气退化的物理机制并不完全相同，例如雾天图像通常与大气散射模型相关，雨线和雪花具有明显的空间结构和运动特征，雨滴会造成局部折射与遮挡，夜间天气还会叠加低照度、噪声和光源眩光。因此，该方向的核心挑战在于：如何在复杂、混合、真实且分布变化明显的天气条件下，恢复视觉质量并保留对下游任务有用的结构信息。

## 2. 研究问题定义

给定一张退化图像 \(I_{weather}\)，目标是学习一个恢复函数：

```text
I_clean = F(I_weather; theta)
```

其中 \(I_clean\) 是清晰图像或更适合视觉分析的增强图像，\(F\) 可以是基于先验的优化模型、深度神经网络、生成模型、扩散模型、物理约束模型，或面向下游任务的联合模型。

常见输入输出设置包括：

| 设置 | 输入 | 输出 | 典型场景 | 主要难点 |
|---|---|---|---|---|
| 单天气恢复 | 单张退化图像 | 单张清晰图像 | 去雾、去雨、去雪 | 退化类型相对明确，但真实退化与合成退化存在差距 |
| 多天气统一恢复 | 多种天气图像 | 统一清晰图像 | All-in-One restoration | 需要区分并处理不同退化模式 |
| 混合天气恢复 | 同一图像含多种退化 | 清晰图像 | 雨雾、雪雾、夜雨 | 退化耦合，单一模型假设不足 |
| 真实场景恢复 | 无配对真实图像 | 感知质量提升图像 | 自动驾驶、监控 | 缺少真实配对标签，评价困难 |
| 任务驱动恢复 | 天气退化图像 + 下游任务 | 检测/分割/识别性能提升 | 自动驾驶感知 | 恢复质量与任务性能不总是一致 |

## 3. 主要子方向

### 3.1 图像去雾 Dehazing

去雾通常关注由大气散射造成的对比度下降、颜色偏移和远处目标模糊。传统方法常利用暗通道先验、颜色衰减先验、透射率估计等物理约束；深度方法则通过 CNN、Transformer、扩散模型或物理模型引导网络学习端到端恢复。

关注问题：

- 合成雾与真实雾之间的 domain gap。
- 远距离区域和天空区域的过增强问题。
- 色彩恢复、边缘保真和深度相关退化建模。

### 3.2 图像去雨 Deraining

去雨任务包括雨线去除、雨 streak 建模、雨层分离和雨滴遮挡恢复。雨线通常具有方向性、尺度变化和透明叠加特征，而真实雨天还可能伴随雾、低照度、反射和运动模糊。

关注问题：

- 雨线与背景纹理的区分。
- 大雨、密集雨、倾斜雨线和动态雨的泛化。
- 去雨后是否损伤细节纹理。

### 3.3 图像去雪 Desnowing

去雪任务需要处理雪花遮挡、半透明雪粒、雪雾和背景颜色偏白等问题。与雨线相比，雪的形状更不规则，尺度变化更大，且容易与高亮纹理混淆。

关注问题：

- 雪花遮挡区域的内容补全。
- 雪雾导致的全局对比度下降。
- 雪与白色背景、道路标线、天空区域的区分。

### 3.4 雨滴去除 Raindrop Removal

雨滴通常附着在镜头或玻璃表面，会产生局部遮挡、折射、模糊和高光。该任务与普通去雨不同，更接近局部遮挡恢复和图像修复问题。

关注问题：

- 雨滴区域检测与背景恢复的联合优化。
- 大面积雨滴遮挡下的结构重建。
- 真实雨滴数据的标注和评价。

### 3.5 低照度与夜间不利天气增强

夜间雨雾、低照度、强光源和噪声往往同时出现。单纯提升亮度可能放大噪声或产生过曝，因此需要同时考虑照明估计、噪声抑制、颜色校正和结构增强。

关注问题：

- 亮度增强与噪声控制的平衡。
- 局部强光、眩光和反射处理。
- 夜间恢复对检测、识别等任务的帮助。

### 3.6 多天气统一恢复 All-in-One Restoration

多天气统一恢复希望用一个模型处理多种天气退化，减少为每类天气单独训练模型的成本。常见思路包括天气类型编码、条件提示、专家混合、动态卷积、注意力选择、prompt learning 和退化感知表示学习。

关注问题：

- 模型是否能识别不同退化类型。
- 多任务训练中是否存在任务冲突。
- 未见过天气组合上的泛化能力。

### 3.7 面向下游任务的恢复

在自动驾驶、遥感和监控中，恢复图像的最终目的可能不是主观好看，而是提升检测、分割、跟踪或识别性能。因此，任务驱动方法会将恢复网络与下游模型联合训练，或采用感知损失、特征一致性损失和任务损失。

关注问题：

- PSNR/SSIM 提升是否对应 mAP/mIoU 提升。
- 恢复模块是否引入伪纹理，干扰下游模型。
- 端到端训练的稳定性和可迁移性。

## 4. 技术路线分类

| 类别 | 核心思想 | 优点 | 局限 | 适合重点阅读的问题 |
|---|---|---|---|---|
| 物理模型/先验方法 | 根据天气退化成像模型或图像统计先验进行恢复 | 可解释性强，训练数据需求低 | 真实复杂场景中假设可能不成立 | 模型假设、先验适用范围、失败案例 |
| CNN 方法 | 通过卷积网络学习局部纹理和退化模式 | 训练稳定，计算相对可控 | 长距离依赖和全局建模能力有限 | 网络结构、残差设计、多尺度融合 |
| GAN 方法 | 通过生成对抗学习提升感知质量 | 视觉效果可能更自然 | 容易产生伪细节，定量指标不稳定 | 感知质量、真实性、伪影控制 |
| Transformer 方法 | 利用自注意力建模长距离依赖 | 全局上下文建模强 | 计算量较高，对数据规模敏感 | 注意力设计、窗口机制、复杂度控制 |
| 扩散模型 | 通过逐步去噪或条件生成恢复图像 | 生成能力强，细节恢复潜力大 | 推理成本高，保真性和可控性需关注 | 采样效率、条件约束、真实感与忠实度 |
| 物理引导深度模型 | 将物理模型参数或先验嵌入深度网络 | 兼顾可解释性与学习能力 | 模型设计复杂，依赖退化假设 | 参数估计、模块解释性、跨场景泛化 |
| 自监督/无监督方法 | 利用无配对数据、循环一致性或退化约束训练 | 更适合真实数据 | 训练目标可能不充分，评价困难 | 无配对学习、真实域适应、稳定性 |
| 多任务/统一模型 | 一个模型处理多种退化或多个视觉任务 | 实用性强，部署成本低 | 任务冲突和性能折中明显 | 退化识别、任务路由、专家机制 |

## 5. 关键挑战

| 挑战 | 说明 | 调研时应关注 |
|---|---|---|
| 合成到真实的差距 | 大量数据集通过合成方式构造，真实天气分布更复杂 | 是否有真实数据验证，是否做 domain adaptation |
| 评价指标不充分 | PSNR/SSIM 偏向像素一致性，未必反映感知质量或任务收益 | 是否同时报告感知指标和下游任务指标 |
| 多天气耦合 | 真实场景常同时存在雨、雾、低光、噪声、眩光 | 是否支持混合退化，是否只在单天气数据上验证 |
| 细节保真 | 过度恢复可能产生伪纹理或改变语义结构 | 是否分析伪影、边缘、纹理和语义一致性 |
| 泛化能力 | 模型可能只适配特定数据集或退化强度 | 是否跨数据集测试，是否做真实图像定性比较 |
| 计算效率 | 部署场景常要求实时或低功耗 | 参数量、FLOPs、推理速度、显存占用 |
| 下游一致性 | 图像看起来更清晰不一定提升检测/分割 | 是否报告 mAP、mIoU、tracking metrics 等 |

## 6. 文献整理

| 论文题目 | 年份 | 会议/期刊| 天气类型 | 方法类别 | 代码/项目链接 |
|---|---:|---|---|---|---|
|[Restoring Vision in Adverse Weather Conditions With Patch-Based Denoising Diffusion Models](https://doi.org/10.1109/TPAMI.2023.3238179) | 2023 | TPAMI | All in one| Diffusion | [code](https://github.com/IGITUGraz/WeatherDiffusion)|
|[Video Adverse-Weather-Component Suppression Network via Weather Messenger and Adversarial Backpropagation](https://doi.org/10.1109/ICCV51070.2023.01214) | 2023 | ICCV | All in one| CNN | [code](https://github.com/scott-yjyang/ViWS-Net)|
|[Adverse Weather Removal with Codebook Priors](https://doi.org/10.1109/ICCV51070.2023.01163) | 2023 | ICCV | All in one| CNN+Transformer | [code](https://github.com/Owen718/AWRCP)|
|[Exploring the Application of Large-Scale Pre-Trained Models on Adverse Weather Removal](https://doi.org/10.1109/TIP.2024.3368961) | 2024 | TIP | All in one| VLM | -|
|[Language-driven All-in-one Adverse Weather Removal](https://doi.org/10.1109/CVPR52733.2024.02352) | 2024 | CVPR | All in one| VLM | -|
|[MOERL: When Mixture-Of-Experts Meet Reinforcement Learning for Adverse Weather Image Restoration](https://doi.org/10.1109/TIP.2024.3501855) | 2024 | ICCV | All in one| VLM | -|
|[MWFormer: Multi-Weather Image Restoration Using Degradation-Aware Transformers](https://doi.org/10.1109/TIP.2024.3501855) | 2024 | TIP | All in one| Transformer | [code](https://github.com/taco-group/MWFormer)|
| [Prompt to Restore, Restore to Prompt: Cyclic Prompting for Universal Adverse Weather Removal](https://doi.org/10.1109/TIP.2025.3627860) | 2025 | TIP  | All in one | VLM | [code](https://github.com/RongxinL/CyclicPrompt) |  
| [Robust Adverse Weather Removal via Spectral-based Spatial Grouping](https://doi.org/10.1109/ICCV51701.2025.01104) | 2025 | ICCV  | All in one | Transformer | [code](https://github.com/jeongyh98/SSGformer) |  
| [All-in-One Transformer for Image Restoration Under Adverse Weather Degradations](https://doi.org/10.1109/TPAMI.2026.3658598) | 2026 | TPAMI  | All in one | CLIP+Transformer | - |  
|  |  |  |  |  |  |  


## 7. 数据集整理

| 数据集名称 | 年份 | 任务 | 天气类型 | 数据类型 | 是否配对 | 图像数量 | 分辨率 | 训练/测试划分 | 合成/真实 | 标注类型 | 下载链接 |
|---|---:|---|---|---|---|---:|---|---|---|---|---|
| DIV2K + Flickr2K | 2017 | 辅助预训练/码本学习 | 无天气退化 | 图像 | 否 | DIV2K: 1,000；Flickr2K: 2,650 | 2K 高质量图像 | DIV2K: 800 train / 100 val / 100 test；Flickr2K 常用于训练 | 真实清晰图像 | clean image | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) / [Flickr2K](https://github.com/limbee/NTIRE2017) |
| Snow100K | 2018 | 去雪 | 雪 | 图像 | 是 | 100K | 最大边长约 640 | 50K train / 50K test；常用 Snow100K-S/M/L 子测试集 | 合成 | clean image、snow mask | [Official](https://sites.google.com/view/yunfuliu/desnownet) |
| Snow100K-real / Snow100K-R | 2018 | 真实去雪评估 | 雪 | 图像 | 否 | 1,329 | 最大边长约 640 | 仅真实测试/定性评估 | 真实 | 无 GT | [Official](https://sites.google.com/view/yunfuliu/desnownet) |
| Rain1200 | 2018 | 去雨 | 雨 | 图像 | 是 | 13.2K | 未说明 | 12,000 train / 1,200 test | 合成 | clean image、rain density | [GCANet](https://github.com/wfs123456/GCANet) |
| RainDrop / Raindrop-A / TestA | 2018 | 雨滴去除 | 雨滴 | 图像 | 是 | 约 1.1K | 未说明 | All-weather 协议常用 818 train / 58 test | 真实雨滴 + 配对 GT | clean image | [Project](https://rui1996.github.io/raindrop/raindrop_removal.html) / [GitHub](https://github.com/rui1996/DeRaindrop) |
| SICE | 2018 | 低照度增强 | 夜间/低照度 | 图像 | 是 | 589 组序列，4,413 张图像 | 高分辨率，具体未说明 | Part1: 360 sequences；Part2: 229 sequences | 真实/多曝光 | reference image | [GitHub](https://github.com/csjcai/SICE) |
| RESIDE / SOTS-outdoor | 2019 | 去雾 | 雾/霾 | 图像 | 是 | 约 87K | 未说明 | 含 ITS、OTS、SOTS、RTTS、HSTS；ARHC 使用 SOTS-outdoor 评估 | 合成 + 真实 | clean image | [Official](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) |
| Outdoor-Rain / Test1 | 2019 | 去雨 + 去雾 | 雨线、雾/霾 | 图像 | 是 | 约 10.5K | 高分辨率，具体未说明 | 常用 8,250 train / 750 test | 合成 | clean image | [TransWeather](https://github.com/jeya-maria-jose/TransWeather) / [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion) |
| SPA-Data | 2019 | 真实去雨 | 雨 | 图像 | 是 | 约 29.5K 或更大规模版本 | 未说明 | CyclicPrompt、AllRestorer 用于真实雨天测试 | 真实 | clean image | [SPANet](https://github.com/stevewongv/SPANet) |
| Dark Zurich | 2019 | 下游语义分割评估 | 夜间/低照度 | 图像 | 否/部分配对 | 8,779 | 未说明 | 含 nighttime、dusk、daytime；用于夜间语义分割 | 真实 | semantic segmentation labels | [MGCDA](https://github.com/sakaridis/MGCDA) |
| NH-HAZE | 2020 | 去雾 | 非均匀雾/霾 | 图像 | 是 | 55 | 未说明 | NTIRE 2020 challenge 数据 | 真实 | clean image | [Official](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/) |
| LOL-v2 | 2020 | 低照度增强 | 低照度/夜间 | 图像 | 是 | Real: 789；Synthetic: 1,000 | 未说明 | Real: 689 train / 100 test；Synthetic: 900 train / 100 test | 真实 + 合成 | clean image | [CVPR-2020-Semi-Low-Light](https://github.com/flyywh/CVPR-2020-Semi-Low-Light/) |
| REVIDE | 2021 | 视频去雾 | 雾 | 视频 | 是 | 48 个视频，1,082 帧 | 未说明 | 42 train / 6 test；928 train frames / 154 test frames | 真实 | clean video frames | [GitHub](https://github.com/BookerDeWitt/REVIDE_Dataset) |
| CSD | 2021 | 去雪 | 雪、雪雾/veil | 图像 | 是 | 10K | 未说明 | MWFormer-real 将其训练集加入训练 | 合成 | clean image | [GitHub](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet) |
| RainDS / RainDS-real | 2021 | 真实去雨、雨滴/雨线恢复 | 雨、雨滴 | 图像 | 是/部分无 GT | 5.8K | 未说明 | 含 rain streak only、raindrop only、混合雨退化；部分真实测试无 GT | 合成 + 真实 | clean image / 无 GT | [GitHub](https://github.com/Songforrr/RainDS_CCN) |
| All-weather / AllWeather | 2022 | 多天气统一恢复 | 雨雾、雪、雨滴 | 图像 | 是 | 18,069 train；测试集约 16K-17K | 未说明 | 训练集由 Snow100K、Outdoor-Rain、RainDrop 组成；测试包含 Test1、RainDrop TestA、Snow100K-L/S | 合成 + 真实雨滴 | clean image | [TransWeather](https://github.com/jeya-maria-jose/TransWeather) |
| RainMotion | 2022 | 视频去雨 | 雨 | 视频 | 是 | 80 个视频，2,800 帧 | 未说明 | 40 train / 40 test；2,000 train frames / 800 test frames | 合成 | clean video frames | [RDD-Net](https://github.com/wangshauitj/RDD-Net) |
| WeatherStream | 2023 | 多天气统一恢复 | 雨、雪、雾 | 视频帧/图像 | 是 | 187,500 | 未说明 | 176,100 train / 11,400 test | 真实/自动采集 | clean image | [WGWS-Net](https://github.com/zhuyr97/WGWS-Net) |
| KITTI-snow | 2023 | 视频去雪 | 雪 | 视频 | 是 | 50 个视频，2,500 帧 | 未说明 | 35 train / 15 test；1,750 train frames / 750 test frames | 合成 | clean video frames | [ViWS-Net](https://github.com/scott-yjyang/ViWS-Net) |
| CDD-11 | 2024 | 多天气统一恢复、复合退化恢复 | 低照度、雾、雨、雪及其复合退化 | 图像 | 是 | 11 类退化，具体数量见仓库 | 1080 x 720 原图；训练 patch 256 x 256 | 官方 train/test | 合成 + 真实 | clean image | [HuggingFace](https://huggingface.co/datasets/gy65896/CDD-11) / [OneRestore](https://github.com/gy65896/OneRestore) |
| HAC | 2025 | 任意混合不利天气恢复 | 雾、雨、雪、夜间、雨滴 | 图像 | 是 | 约 316K | 未说明 | 含 31 种退化组合；训练集自动生成，测试集人工调制 | 合成 | clean image、退化类型标签 | [GitHub](https://github.com/arun-kollan/RAHC) / [Paper](https://arxiv.org/abs/2305.09996) |
| WeatherBench | 2025 | 真实多天气统一恢复 | 雨、雪、雾、昼夜光照 | 图像 | 是 | 42,002 对 | 512 x 512 | 41,402 train / 600 test | 真实 | clean image | [GitHub](https://github.com/guanqiyuan/WeatherBench) |

常见记录维度建议：

| 维度 | 说明 |
|---|---|
| 数据来源 | 室外真实采集、室内合成、仿真平台、网络收集 |
| 退化强度 | 轻度、中度、重度，是否有连续强度标注 |
| 场景类型 | 城市道路、自然场景、遥感、监控、室内、夜间 |
| 标签质量 | 是否人工标注，是否存在错配或不对齐 |
| 真实性 | 天气退化是否符合真实物理分布 |
| 任务适配 | 是否适合恢复、检测、分割或多任务评估 |

## 9. 评价指标整理模板

| 指标 | 类型 | 计算对象 | 是否需要 GT | 越大越好/越小越好 | 反映能力 | 局限 | 适用任务 | 备注 |
|---|---|---|---|---|---|---|---|---|
| PSNR | 像素级 | 恢复图像 vs GT | 是 | 越大越好 | 像素误差 | 与主观视觉质量不完全一致 | 配对数据恢复 |  |
| SSIM | 结构相似性 | 恢复图像 vs GT | 是 | 越大越好 | 结构保持 | 对感知真实性有限 | 配对数据恢复 |  |
| LPIPS | 感知距离 | 恢复图像 vs GT | 是 | 越小越好 | 深度特征感知差异 | 依赖预训练特征 | 感知质量评价 |  |
| NIQE | 无参考质量 | 单张恢复图像 | 否 | 越小越好 | 自然图像统计 | 不一定适配所有天气场景 | 真实无配对数据 |  |
| BRISQUE | 无参考质量 | 单张恢复图像 | 否 | 越小越好 | 自然统计失真 | 与人类偏好不总一致 | 真实图像评价 |  |
| FID | 分布距离 | 恢复图像集 vs 真实清晰图像集 | 否/弱依赖 | 越小越好 | 分布真实性 | 不评价单图结构忠实度 | 生成式恢复 |  |
| mAP | 下游检测 | 检测结果 vs 标注 | 是 | 越大越好 | 检测性能 | 依赖检测器和标注 | 自动驾驶/监控 |  |
| mIoU | 下游分割 | 分割结果 vs 标注 | 是 | 越大越好 | 语义分割性能 | 依赖分割模型和数据标注 | 场景理解 |  |
| FPS | 效率 | 推理速度 | 否 | 越大越好 | 实时性 | 与硬件强相关 | 部署评估 | 需记录 GPU/CPU |
| Params/FLOPs | 模型复杂度 | 网络结构 | 否 | 通常越小越好 | 计算和存储成本 | 不完全等价于真实速度 | 模型比较 |  |
