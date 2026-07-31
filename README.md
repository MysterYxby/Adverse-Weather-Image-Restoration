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

> 2025-2026 年论文按顶刊/顶会补全：期刊限 TIP、TPAMI；会议限 CCF-A 类会议。代码不再作为收录条件，未开源或暂未检索到代码的条目统一标注为 `-`。

### 6.1 去雾 Dehazing

| 论文题目 | 年份 | 会议/期刊| 天气类型 | 方法类别 | 代码/项目链接 |
|---|---:|---|---|---|---|
| [Deep Multi-Model Fusion for Single-Image Dehazing](https://openaccess.thecvf.com/content_ICCV_2019/html/Deng_Deep_Multi-Model_Fusion_for_Single-Image_Dehazing_ICCV_2019_paper.html) | 2019 | ICCV | 去雾 | Multi-model Fusion | [code](https://github.com/zijundeng/DM2F-Net) |
| [Multi-Scale Boosted Dehazing Network with Dense Feature Fusion](https://openaccess.thecvf.com/content_CVPR_2020/html/Dong_Multi-Scale_Boosted_Dehazing_Network_With_Dense_Feature_Fusion_CVPR_2020_paper.html) | 2020 | CVPR | 去雾 | CNN | [code](https://github.com/BookerDeWitt/MSBDN-DFF) |
| [PSD: Principled Synthetic-to-Real Dehazing Guided by Physical Priors](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_PSD_Principled_Synthetic-to-Real_Dehazing_Guided_by_Physical_Priors_CVPR_2021_paper.html) | 2021 | CVPR | 去雾 | Physical Priors + Domain Adaptation | [code](https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors) |
| [Vision Transformers for Single Image Dehazing](https://doi.org/10.1109/TIP.2023.3256763) | 2023 | TIP | 去雾 | Transformer | [code](https://github.com/IDKiro/DehazeFormer) |
| [Curricular Contrastive Regularization for Physics-aware Single Image Dehazing](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Curricular_Contrastive_Regularization_for_Physics-Aware_Single_Image_Dehazing_CVPR_2023_paper.html) | 2023 | CVPR | 去雾 | Physics-aware Contrastive Learning | [code](https://github.com/yuzheng9/C2PNet) |
| [Revitalizing Real Image Dehazing via High-Quality Codebook Priors](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_RIDCP_Revitalizing_Real_Image_Dehazing_via_High-Quality_Codebook_Priors_CVPR_2023_paper.html) | 2023 | CVPR | 去雾 | Codebook Priors | [code](https://github.com/RQ-Wu/RIDCP_dehazing) |
| [Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing](https://openaccess.thecvf.com/content/CVPR2025/html/Fu_Iterative_Predictor-Critic_Code_Decoding_for_Real-World_Image_Dehazing_CVPR_2025_paper.html) | 2025 | CVPR | 去雾 | Predictor-Critic Code Decoding | - |
| [Learning Hazing to Dehazing: Towards Realistic Haze Generation for Real-World Image Dehazing](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Learning_Hazing_to_Dehazing_Towards_Realistic_Haze_Generation_for_Real-World_CVPR_2025_paper.html) | 2025 | CVPR | 去雾 | Haze Generation + Dehazing | - |
| [Tokenize Image Patches: Global Context Fusion for Effective Haze Removal in Large Images](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Tokenize_Image_Patches_Global_Context_Fusion_for_Effective_Haze_Removal_CVPR_2025_paper.html) | 2025 | CVPR | 去雾 | Tokenization + Global Context Fusion | - |
| [CoA: Towards Real Image Dehazing via Compression-and-Adaptation](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_CoA_Towards_Real_Image_Dehazing_via_Compression-and-Adaptation_CVPR_2025_paper.html) | 2025 | CVPR | 去雾 | Compression-and-Adaptation | - |
| [PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing](https://openaccess.thecvf.com/content/ICCV2025/html/Tsai_PHATNet_A_Physics-guided_Haze_Transfer_Network_for_Domain-adaptive_Real-world_Image_ICCV_2025_paper.html) | 2025 | ICCV | 去雾 | Physics-guided Haze Transfer | - |
| [GenHaze: Pioneering Controllable One-Step Realistic Haze Generation for Real-World Dehazing](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_GenHaze_Pioneering_Controllable_One-Step_Realistic_Haze_Generation_for_Real-World_Dehazing_ICCV_2025_paper.html) | 2025 | ICCV | 去雾 | Controllable Haze Generation | - |
| [Frequency Domain-Based Diffusion Model for Unpaired Image Dehazing](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_Frequency_Domain-Based_Diffusion_Model_for_Unpaired_Image_Dehazing_ICCV_2025_paper.html) | 2025 | ICCV | 去雾 | Frequency-domain Diffusion | - |
| [HazeFlow: Revisit Haze Physical Model as ODE and Non-Homogeneous Haze Generation for Real-World Dehazing](https://openaccess.thecvf.com/content/ICCV2025/html/Shin_HazeFlow_Revisit_Haze_Physical_Model_as_ODE_and_Non-Homogeneous_Haze_ICCV_2025_paper.html) | 2025 | ICCV | 去雾 | Physical ODE + Haze Generation | - |
| [PDDA: Prompt-driven Domain Adaptation for Real-world Image Dehazing](https://github.com/YanZhang-zy/PDDA) | 2026 | TIP | 去雾 | Prompt-driven Domain Adaptation | [code](https://github.com/YanZhang-zy/PDDA) |
| [Bilevel Layer-Positioning LoRA for Real Image Dehazing](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Bilevel_Layer-Positioning_LoRA_for_Real_Image_Dehazing_CVPR_2026_paper.html) | 2026 | CVPR | 去雾 | LoRA + Bilevel Optimization | [code](https://github.com/YanZhang-zy/BiLaLoRA) |
| [From Events to Clarity: The Event-Guided Diffusion Framework for Dehazing](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_From_Events_to_Clarity_The_Event-Guided_Diffusion_Framework_for_Dehazing_CVPR_2026_paper.html) | 2026 | CVPR | 去雾 | Event-guided Diffusion | [code](https://github.com/DavisWANG0/EvDehaze) |
| [Disentanglement-wise Image Dehazing through Cross-Domain Manifold Consensus](https://openaccess.thecvf.com/content/CVPR2026/html/Lyu_Disentanglement-wise_Image_Dehazing_through_Cross-Domain_Manifold_Consensus_CVPR_2026_paper.html) | 2026 | CVPR | 去雾 | Cross-domain Manifold Consensus | - |
| [Adaptive Dynamic Dehazing via Instruction-Driven and Task-Feedback Closed-Loop Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/38287) | 2026 | AAAI | 去雾 | Instruction-driven Closed-loop Optimization | - |
| [DehazeGS: Seeing Through Fog with 3D Gaussian Splatting](https://ojs.aaai.org/index.php/AAAI/article/view/38205) | 2026 | AAAI | 去雾/3D 场景 | 3D Gaussian Splatting | - |

### 6.2 去雨与雨滴去除 Deraining / Raindrop Removal

| 论文题目 | 年份 | 会议/期刊| 天气类型 | 方法类别 | 代码/项目链接 |
|---|---:|---|---|---|---|
| [Removing Rain from Single Images via a Deep Detail Network](https://openaccess.thecvf.com/content_cvpr_2017/html/Fu_Removing_Rain_From_CVPR_2017_paper.html) | 2017 | CVPR | 去雨 | CNN | [code](https://github.com/XMU-smartdsp/Removing_Rain) |
| [Attentive Generative Adversarial Network for Raindrop Removal from A Single Image](https://openaccess.thecvf.com/content_cvpr_2018/html/Qian_Attentive_Generative_Adversarial_CVPR_2018_paper.html) | 2018 | CVPR | 雨滴去除 | Attention + GAN | [code](https://github.com/rui1996/DeRaindrop) |
| [Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Spatial_Attentive_Single-Image_Deraining_With_a_High_Quality_Real_Rain_Dataset_CVPR_2019_paper.html) | 2019 | CVPR | 去雨 | CNN + Spatial Attention | [code](https://github.com/stevewongv/SPANet) |
| [Removing Raindrops and Rain Streaks in One Go](https://openaccess.thecvf.com/content/CVPR2021/html/Quan_Removing_Raindrops_and_Rain_Streaks_in_One_Go_CVPR_2021_paper.html) | 2021 | CVPR | 去雨滴 + 去雨线 | CNN | [code](https://github.com/Songforrr/RainDS_CCN) |
| [Image De-raining Transformer](https://openaccess.thecvf.com/content/CVPR2022/html/Xiao_Image_De-Raining_Transformer_CVPR_2022_paper.html) | 2022 | CVPR | 去雨 | Transformer | [code](https://github.com/jiexiaou/IDT) |
| [Learning A Sparse Transformer Network for Effective Image Deraining](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Learning_a_Sparse_Transformer_Network_for_Effective_Image_Deraining_CVPR_2023_paper.html) | 2023 | CVPR | 去雨 | Sparse Transformer | [code](https://github.com/cschenxiang/DRSformer) |
| [Sparse Sampling Transformer with Uncertainty-Driven Ranking for Unified Removal of Raindrops and Rain Streaks](https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Sparse_Sampling_Transformer_with_Uncertainty-Driven_Ranking_for_Unified_Removal_of_ICCV_2023_paper.html) | 2023 | ICCV | 去雨滴 + 去雨线 | Sparse Sampling Transformer | [code](https://github.com/Ephemeral182/UDR-S2Former_deraining) |
| [Semi-Supervised State-Space Model with Dynamic Stacking Filter for Real-World Video Deraining](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_Semi-Supervised_State-Space_Model_with_Dynamic_Stacking_Filter_for_Real-World_Video_CVPR_2025_paper.html) | 2025 | CVPR | 视频去雨 | State Space Model | - |
| [Rethinking Nighttime Image Deraining via Learnable Color Space and Robust Imaging Priors](https://openreview.net/forum?id=MQJGqVeDd4) | 2025 | NeurIPS | 夜间去雨 | Learnable Color Space + Imaging Priors | - |
| [PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining](https://openaccess.thecvf.com/content/ICCV2025/html/Ruan_PRE-Mamba_A_4D_State_Space_Model_for_Ultra-High-Frequent_Event_Camera_ICCV_2025_paper.html) | 2025 | ICCV | Event Camera 去雨 | 4D State Space Model | - |
| [UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_UniRain_Unified_Image_Deraining_with_RAG-based_Dataset_Distillation_and_Multi-objective_CVPR_2026_paper.html) | 2026 | CVPR | 去雨、雨滴、夜间雨 | RAG Dataset Distillation + MoE | [code](https://github.com/QianfengY/UniRain) |
| [Unpaired Image Deraining Using Reward-Guided Self-Reinforcement Strategy](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Unpaired_Image_Deraining_Using_Reward-Guided_Self-Reinforcement_Strategy_CVPR_2026_paper.html) | 2026 | CVPR | 去雨 | Reward-guided Self-Reinforcement | - |
| [Resolving High-Frequency Conflicts in Deraining and Super-Resolution via Ambidextrous Learning](https://ojs.aaai.org/index.php/AAAI/article/view/37575) | 2026 | AAAI | 去雨 + 超分 | Ambidextrous Learning | - |

### 6.3 去雪 Desnowing

| 论文题目 | 年份 | 会议/期刊| 天气类型 | 方法类别 | 代码/项目链接 |
|---|---:|---|---|---|---|
| [DesnowNet: Context-Aware Deep Network for Snow Removal](https://doi.org/10.1109/TIP.2018.2861572) | 2018 | TIP | 去雪 | CNN | [code](https://github.com/linYDTHU/DesnowNet_Context-Aware_Deep_Network_for_Snow_Removal) |
| [JSTASR: Joint Size and Transparency-Aware Snow Removal Algorithm Based on Modified Partial Convolution and Veiling Effect Removal](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5485_ECCV_2020_paper.php) | 2020 | ECCV | 去雪 | Partial Convolution + Veiling Removal | [code](https://github.com/weitingchen83/JSTASR-DesnowNet-ECCV-2020) |
| [All Snow Removed: Single Image Desnowing Algorithm Using Hierarchical Dual-tree Complex Wavelet Representation and Contradict Channel Loss](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_All_Snow_Removed_Single_Image_Desnowing_Algorithm_Using_Hierarchical_Dual-Tree_Complex_ICCV_2021_paper.html) | 2021 | ICCV | 去雪 | CNN + Wavelet | [code](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet) |
| [SnowMaster: Comprehensive Real-world Image Desnowing via MLLM with Multi-Model Feedback Optimization](https://openaccess.thecvf.com/content/CVPR2025/html/Lai_SnowMaster_Comprehensive_Real-world_Image_Desnowing_via_MLLM_with_Multi-Model_Feedback_CVPR_2025_paper.html) | 2025 | CVPR | 去雪 | MLLM + Multi-model Feedback | - |
| [Wavelet-Enhanced Desnowing: A Novel Single Image Restoration Framework](https://ieeexplore.ieee.org/document/11104149) | 2025 | TIP | 去雪 | Wavelet-enhanced Restoration | - |

### 6.4 低照度与夜间恢复 Low-Light / Nighttime Restoration

| 论文题目 | 年份 | 会议/期刊| 天气类型 | 方法类别 | 代码/项目链接 |
|---|---:|---|---|---|---|
| [Low-Light Image Enhancement with Normalizing Flow](https://ojs.aaai.org/index.php/AAAI/article/view/19923) | 2022 | AAAI | 低照度 | Normalizing Flow | [code](https://github.com/wyf0912/LLFlow) |
| [SNR-Aware Low-Light Image Enhancement](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_SNR-Aware_Low-Light_Image_Enhancement_CVPR_2022_paper.html) | 2022 | CVPR | 低照度 | SNR-aware Transformer/CNN | [code](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance) |
| [Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement](https://doi.org/10.1109/ICCV51070.2023.01150) | 2023 | ICCV | 低照度 | Retinex + Transformer | [code](https://github.com/caiyuanhao1998/Retinexformer) |
| [Thermal-Aware Low-Light Image Enhancement: A Real-World Dataset and Baseline](https://ojs.aaai.org/index.php/AAAI/article/view/32887) | 2025 | AAAI | 低照度 | Thermal-aware LLIE | - |
| [Zero-Shot Low-Light Image Enhancement via Latent Diffusion Models](https://ojs.aaai.org/index.php/AAAI/article/view/32398) | 2025 | AAAI | 低照度 | Latent Diffusion | - |
| [IniRetinex: Rethinking Retinex-type Low-Light Image Enhancer via Initial Decomposition](https://ojs.aaai.org/index.php/AAAI/article/view/32289) | 2025 | AAAI | 低照度 | Retinex | - |
| [URWKV: Unified RWKV Model with Multi-state Perspective for Low-light Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_URWKV_Unified_RWKV_Model_with_Multi-state_Perspective_for_Low-light_Image_CVPR_2025_paper.html) | 2025 | CVPR | 低照度 | RWKV | - |
| [Efficient Diffusion as Low Light Enhancer](https://openaccess.thecvf.com/content/CVPR2025/html/Lan_Efficient_Diffusion_as_Low_Light_Enhancer_CVPR_2025_paper.html) | 2025 | CVPR | 低照度 | Efficient Diffusion | - |
| [Improving Visual and Downstream Performance of Low-Light Enhancer with Vision Foundation Models Collaboration](https://openaccess.thecvf.com/content/CVPR2025/html/Gu_Improving_Visual_and_Downstream_Performance_of_Low-Light_Enhancer_with_Vision_CVPR_2025_paper.html) | 2025 | CVPR | 低照度 | Vision Foundation Model Collaboration | - |
| [HVI: A New Color Space for Low-light Image Enhancement](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.html) | 2025 | CVPR | 低照度 | Color Space + CNN | [code](https://github.com/Fediory/HVI-CIDNet) |
| [DarkIR: Robust Low-Light Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Feijoo_DarkIR_Robust_Low-Light_Image_Restoration_CVPR_2025_paper.html) | 2025 | CVPR | 低照度、噪声、模糊 | Low-light Restoration | [code](https://github.com/cidautai/DarkIR) |
| [CWNet: Causal Wavelet Network for Low-Light Image Enhancement](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_CWNet_Causal_Wavelet_Network_for_Low-Light_Image_Enhancement_ICCV_2025_paper.html) | 2025 | ICCV | 低照度 | Causal Wavelet Network | - |
| [RetinexMCNet: A Memory Controller Dominated Network for Low-Light Video Enhancement Based on Retinex](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_RetinexMCNet_A_Memory_Controller_Dominated_Network_for_Low-Light_Video_Enhancement_ICCV_2025_paper.html) | 2025 | ICCV | 低照度视频 | Retinex + Memory Controller | - |
| [GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts](https://openaccess.thecvf.com/content/ICCV2025/html/Liao_GM-MoE_Low-Light_Enhancement_with_Gated-Mechanism_Mixture-of-Experts_ICCV_2025_paper.html) | 2025 | ICCV | 低照度 | Gated MoE | - |
| [Low-Light Image Enhancement Using Event-Based Illumination Estimation](https://openaccess.thecvf.com/content/ICCV2025/html/Sun_Low-Light_Image_Enhancement_Using_Event-Based_Illumination_Estimation_ICCV_2025_paper.html) | 2025 | ICCV | 低照度 | Event-based Illumination Estimation | - |
| [Spike-RetinexFormer: Rethinking Low-light Image Enhancement with Spike Transformer](https://openreview.net/forum?id=8W8SRZIpJP) | 2025 | NeurIPS | 低照度 | Spike Transformer + Retinex | - |
| [ZeroIDIR: Zero-Reference Illumination Degradation Image Restoration with Perturbed Consistency Diffusion Models](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_ZeroIDIR_Zero-Reference_Illumination_Degradation_Image_Restoration_with_Perturbed_Consistency_Diffusion_CVPR_2026_paper.html) | 2026 | CVPR | 低照度、背光、曝光退化 | Perturbed Consistency Diffusion | [code](https://github.com/JianghaiSCU/ZeroIDIR) |
| [Event-Illumination Collaborative Low-light Image Enhancement with a High-resolution Real-world Dataset](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Event-Illumination_Collaborative_Low-light_Image_Enhancement_with_a_High-resolution_Real-world_Dataset_CVPR_2026_paper.html) | 2026 | CVPR | 低照度 | Event-Illumination Collaboration | - |
| [MR. Illuminate: Zero-Shot Low-Light Image Enhancement with Diffusion Prior](https://openaccess.thecvf.com/content/CVPR2026/html/Cho_MR._Illuminate_Zero-Shot_Low-Light_Image_Enhancement_with_Diffusion_Prior_CVPR_2026_paper.html) | 2026 | CVPR | 低照度 | Diffusion Prior | - |
| [Bi-Bridge: Bidirectional Diffusion Bridges for Low-Light Image Enhancement](https://openaccess.thecvf.com/content/CVPR2026/html/Hua_Bi-Bridge_Bidirectional_Diffusion_Bridges_for_Low-Light_Image_Enhancement_CVPR_2026_paper.html) | 2026 | CVPR | 低照度 | Bidirectional Diffusion Bridge | - |
| [Multinex: Lightweight Low-light Image Enhancement via Multi-prior Retinex](https://openaccess.thecvf.com/content/CVPR2026/html/Brateanu_Multinex_Lightweight_Low-light_Image_Enhancement_via_Multi-prior_Retinex_CVPR_2026_paper.html) | 2026 | CVPR | 低照度 | Multi-prior Retinex | - |
| [BiEvLight: Bi-level Learning of Task-Aware Event Refinement for Low-Light Image Enhancement](https://openaccess.thecvf.com/content/CVPR2026/html/Yao_BiEvLight_Bi-level_Learning_of_Task-Aware_Event_Refinement_for_Low-Light_Image_CVPR_2026_paper.html) | 2026 | CVPR | 低照度 | Event Refinement | - |
| [eRetinexGS: Retinex Modeling for Low-Light Scene Enhancement via Event Streams and 3D Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_eRetinexGS_Retinex_Modeling_for_Low-Light_Scene_Enhancement_via_Event_Streams_CVPR_2026_paper.html) | 2026 | CVPR | 低照度/3D 场景 | Event + Retinex + 3DGS | - |
| [Fine-Grained Detail Preservation in Extreme Dark Image Restoration](https://ojs.aaai.org/index.php/AAAI/article/view/38276) | 2026 | AAAI | 极暗图像恢复 | Detail Preservation | - |
| [ICLR: Inter-Chrominance and Luminance Interaction for Natural Low-Light Image Enhancement](https://ojs.aaai.org/index.php/AAAI/article/view/38123) | 2026 | AAAI | 低照度 | Chrominance-Luminance Interaction | - |

### 6.5 All-in-One / 多天气统一恢复

| 论文题目 | 年份 | 会议/期刊| 天气类型 | 方法类别 | 代码/项目链接 |
|---|---:|---|---|---|---|
| [TransWeather: Transformer-Based Restoration of Images Degraded by Adverse Weather Conditions](https://openaccess.thecvf.com/content/CVPR2022/html/Valanarasu_TransWeather_Transformer-Based_Restoration_of_Images_Degraded_by_Adverse_Weather_Conditions_CVPR_2022_paper.html) | 2022 | CVPR | All in one | Transformer | [code](https://github.com/jeya-maria-jose/TransWeather) |
| [Learning Multiple Adverse Weather Removal via Two-stage Knowledge Learning and Multi-contrastive Regularization](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Learning_Multiple_Adverse_Weather_Removal_via_Two-Stage_Knowledge_Learning_and_Multi-Contrastive_CVPR_2022_paper.html) | 2022 | CVPR | All in one | Knowledge Distillation + Contrastive Learning | [code](https://github.com/fingerk28/Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal) |
| [All-in-One Image Restoration for Unknown Corruption](https://openaccess.thecvf.com/content/CVPR2022/html/Li_All-in-One_Image_Restoration_for_Unknown_Corruption_CVPR_2022_paper.html) | 2022 | CVPR | 通用 All in one | CNN + Contrastive Learning | [code](https://github.com/XLearning-SCU/2022-CVPR-AirNet) |
| [Restoring Vision in Adverse Weather Conditions With Patch-Based Denoising Diffusion Models](https://doi.org/10.1109/TPAMI.2023.3238179) | 2023 | TPAMI | All in one | Diffusion | [code](https://github.com/IGITUGraz/WeatherDiffusion) |
| [Adverse Weather Removal with Codebook Priors](https://doi.org/10.1109/ICCV51070.2023.01163) | 2023 | ICCV | All in one | CNN + Transformer + Codebook | [code](https://github.com/Owen718/AWRCP) |
| [Learning Weather-General and Weather-Specific Features for Image Restoration Under Multiple Adverse Weather Conditions](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_Learning_Weather-General_and_Weather-Specific_Features_for_Image_Restoration_Under_Multiple_Adverse_CVPR_2023_paper.html) | 2023 | CVPR | All in one | Weather-general/specific Learning | [code](https://github.com/zhuyr97/WGWS-Net) |
| [PromptIR: Prompting for All-in-One Blind Image Restoration](https://openreview.net/forum?id=KAlSIL4tXU) | 2023 | NeurIPS | 通用 All in one | Prompt Learning | [code](https://github.com/va1shn9v/PromptIR) |
| [Language-driven All-in-one Adverse Weather Removal](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Language-driven_All-in-one_Adverse_Weather_Removal_CVPR_2024_paper.html) | 2024 | CVPR | All in one | VLM + Language Prompt | [code](https://github.com/noxsine/LDR) |
| [Controlling Vision-Language Models for Universal Image Restoration](https://openreview.net/forum?id=t3vnnLeajU) | 2024 | ICLR | 通用 All in one | DA-CLIP + Diffusion | [code](https://github.com/Algolzw/daclip-uir) |
| [Selective Hourglass Mapping for Universal Image Restoration Based on Diffusion Model](https://openaccess.thecvf.com/content/CVPR2024/html/Zheng_Selective_Hourglass_Mapping_for_Universal_Image_Restoration_Based_on_Diffusion_Model_CVPR_2024_paper.html) | 2024 | CVPR | 通用 All in one | Diffusion | [code](https://github.com/iSEE-Laboratory/DiffUIR) |
| [InstructIR: High-Quality Image Restoration Following Human Instructions](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5238_ECCV_2024_paper.php) | 2024 | ECCV | 通用 All in one | Instruction/VLM | [code](https://github.com/mv-lab/InstructIR) |
| [Restore Anything with Masks](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1336_ECCV_2024_paper.php) | 2024 | ECCV | 通用 All in one | Mask Image Modeling | [code](https://github.com/DragonisCV/RAM) |
| [Restoring Images in Adverse Weather Conditions via Histogram Transformer](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6366_ECCV_2024_paper.php) | 2024 | ECCV | All in one | Histogram Transformer | [code](https://github.com/sunshangquan/Histoformer) |
| [Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2698_ECCV_2024_paper.php) | 2024 | ECCV | All in one | Diffusion + Prompt Pool | [code](https://github.com/Ephemeral182/ECCV24_T3-DiffWeather) |
| [Neural Degradation Representation Learning for All-In-One Image Restoration](https://doi.org/10.1109/TIP.2024.3425893) | 2024 | TIP | 通用 All in one | Degradation Representation | [code](https://github.com/mdyao/NDR-Restore) |
| [MWFormer: Multi-Weather Image Restoration Using Degradation-Aware Transformers](https://doi.org/10.1109/TIP.2024.3501855) | 2024 | TIP | All in one | Degradation-aware Transformer | [code](https://github.com/taco-group/MWFormer) |
| [Efficient Deweather Mixture-of-Experts with Uncertainty-aware Feature-wise Linear Modulation](https://ojs.aaai.org/index.php/AAAI/article/view/28030) | 2024 | AAAI | All in one | MoE + Uncertainty | [code](https://github.com/RoyZry98/MoFME-Pytorch) |
| [Prompt to Restore, Restore to Prompt: Cyclic Prompting for Universal Adverse Weather Removal](https://doi.org/10.1109/TIP.2025.3627860) | 2025 | TIP | All in one | VLM + Cyclic Prompt | [code](https://github.com/RongxinL/CyclicPrompt) |
| [Learning Dynamic Prompts in Latent Space for All-in-One Image Restoration](https://doi.org/10.1109/TIP.2025.3557676) | 2025 | TIP | 通用 All in one | Dynamic Prompt Learning | - |
| [Perceive-IR: Learning to Perceive Degradation Better for All-in-One Image Restoration](https://doi.org/10.1109/TIP.2025.3557678) | 2025 | TIP | 通用 All in one | Degradation Perception | - |
| [ReviveDiff: A Universal Diffusion Model for Image Recovery Against Weather Degradations](https://doi.org/10.1109/TIP.2025.3566808) | 2025 | TIP | All in one | Diffusion | - |
| [Debiased All-in-one Image Restoration with Task Uncertainty Regularization](https://ojs.aaai.org/index.php/AAAI/article/view/32905) | 2025 | AAAI | 通用 All in one | Task Uncertainty Regularization | [code](https://github.com/Aitical/TUR) |
| [All-Weather Visual In-Context Learning for Remote Sensing Image Restoration](https://ojs.aaai.org/index.php/AAAI/article/view/32632) | 2025 | AAAI | 遥感 All in one | Visual In-Context Learning | - |
| [When Unrolling Meets Prompts: A New Paradigm of Adaptive Prompt Learning for All-in-One Image Restoration](https://ojs.aaai.org/index.php/AAAI/article/view/32609) | 2025 | AAAI | 通用 All in one | Unrolling + Prompt Learning | - |
| [MPMF: Making Large Multimodal Model A Strong Fine-Grained Fixer](https://ojs.aaai.org/index.php/AAAI/article/view/32737) | 2025 | AAAI | 通用图像修复/恢复 | MLLM Fine-grained Fixing | - |
| [PPTformer: Pseudo Multi-Prompt Transformer for Blind Super-Resolution Image Restoration](https://ojs.aaai.org/index.php/AAAI/article/view/32492) | 2025 | AAAI | 盲图像恢复 | Pseudo Multi-Prompt Transformer | - |
| [UniRestore: Unified Perceptual and Task-Oriented Image Restoration Model Using Diffusion Prior](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_UniRestore_Unified_Perceptual_and_Task-Oriented_Image_Restoration_Model_Using_Diffusion_CVPR_2025_paper.html) | 2025 | CVPR | 通用 All in one | Diffusion Prior | - |
| [Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Luo_Visual-Instructed_Degradation_Diffusion_for_All-in-One_Image_Restoration_CVPR_2025_paper.html) | 2025 | CVPR | 通用 All in one | Visual Instruction + Diffusion | - |
| [Continuous Adverse Weather Removal via Degradation-Aware Distillation](https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Continuous_Adverse_Weather_Removal_via_Degradation-Aware_Distillation_CVPR_2025_paper.html) | 2025 | CVPR | All in one | Degradation-aware Distillation | - |
| [UHD-processer: Unified UHD Image Restoration with Progressive Frequency Learning and Degradation-aware Prompts](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_UHD-processer_Unified_UHD_Image_Restoration_with_Progressive_Frequency_Learning_and_CVPR_2025_paper.html) | 2025 | CVPR | UHD All in one | Frequency Learning + Degradation Prompt | - |
| [Complexity Experts are Task-Discriminative Learners for Any Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Zamfir_Complexity_Experts_are_Task-Discriminative_Learners_for_Any_Image_Restoration_CVPR_2025_paper.html) | 2025 | CVPR | 通用 All in one | MoE | [code](https://github.com/eduardzamfir/MoCE-IR) |
| [Degradation-Aware Feature Perturbation for All-in-One Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Tian_Degradation-Aware_Feature_Perturbation_for_All-in-One_Image_Restoration_CVPR_2025_paper.html) | 2025 | CVPR | 通用 All in one | Feature Perturbation | [code](https://github.com/TxpHome/DFPIR) |
| [Vision-Language Gradient Descent-driven All-in-One Deep Unfolding Networks](https://openaccess.thecvf.com/content/CVPR2025/html/Zeng_Vision-Language_Gradient_Descent-driven_All-in-One_Deep_Unfolding_Networks_CVPR_2025_paper.html) | 2025 | CVPR | 通用 All in one | VLM + Deep Unfolding | [code](https://github.com/xianggkl/VLU-Net) |
| [GenDeg: Diffusion-Based Degradation Synthesis for Generalizable All-In-One Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Rajagopalan_GenDeg_Diffusion-Based_Degradation_Synthesis_for_Generalizable_All-In-One_Image_Restoration_CVPR_2025_paper.html) | 2025 | CVPR | 通用 All in one | Diffusion Data Synthesis | [code](https://github.com/sudraj2002/GenDeg) |
| [Dual-level Prototype Learning for Composite Degraded Image Restoration](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Dual-level_Prototype_Learning_for_Composite_Degraded_Image_Restoration_ICCV_2025_paper.html) | 2025 | ICCV | 复合退化恢复 | Prototype Learning | - |
| [Robust Adverse Weather Removal via Spectral-based Spatial Grouping](https://openaccess.thecvf.com/content/ICCV2025/html/Jeong_Robust_Adverse_Weather_Removal_via_Spectral-based_Spatial_Grouping_ICCV_2025_paper.html) | 2025 | ICCV | All in one | Spectral-based Spatial Grouping | [code](https://github.com/jeongyh98/SSGformer) |
| [UniRes: Universal Image Restoration for Complex Degradations](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_UniRes_Universal_Image_Restoration_for_Complex_Degradations_ICCV_2025_paper.html) | 2025 | ICCV | 通用 All in one | Universal Restoration | - |
| [LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling](https://openaccess.thecvf.com/content/ICCV2025/html/Li_LD-RPS_Zero-Shot_Unified_Image_Restoration_via_Latent_Diffusion_Recurrent_Posterior_ICCV_2025_paper.html) | 2025 | ICCV | 通用 All in one | Latent Diffusion Posterior Sampling | - |
| [MOERL: When Mixture-of-Experts Meet Reinforcement Learning for Adverse Weather Image Restoration](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_MOERL_When_Mixture-of-Experts_Meet_Reinforcement_Learning_for_Adverse_Weather_Image_ICCV_2025_paper.html) | 2025 | ICCV | All in one | MoE + Reinforcement Learning | - |
| [Controllable Weather Synthesis and Removal with Video Diffusion Models](https://openaccess.thecvf.com/content/ICCV2025/html/Lin_Controllable_Weather_Synthesis_and_Removal_with_Video_Diffusion_Models_ICCV_2025_paper.html) | 2025 | ICCV | 视频天气合成与去除 | Video Diffusion | - |
| [Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](https://openaccess.thecvf.com/content/ICCV2025/html/Yang_Towards_a_Universal_Image_Degradation_Model_via_Content-Degradation_Disentanglement_ICCV_2025_paper.html) | 2025 | ICCV | 通用退化建模 | Content-Degradation Disentanglement | - |
| [MIORe & VAR-MIORe: Benchmarks to Push the Boundaries of Restoration](https://openaccess.thecvf.com/content/ICCV2025/html/Ciubotariu_MIORe__VAR-MIORe_Benchmarks_to_Push_the_Boundaries_of_Restoration_ICCV_2025_paper.html) | 2025 | ICCV | 通用图像恢复 Benchmark | Benchmark + VAR | - |
| [Real-World Adverse Weather Image Restoration via Dual-Level Degradation-Aware Multiple Prompting](https://openreview.net/forum?id=qK2DaGhWg8) | 2025 | NeurIPS | All in one | Degradation-aware Multiple Prompting | - |
| [Bio-Inspired Image Restoration Foundation Model](https://openreview.net/forum?id=JKtXbb9NsH) | 2025 | NeurIPS | 通用图像恢复基础模型 | Bio-inspired Foundation Model | - |
| [InstructRestore: Region-customizable Image Restoration with Human Instructions](https://openreview.net/forum?id=daMqQCb5tQ) | 2025 | NeurIPS | 指令式图像恢复 | Human Instruction | - |
| [Kernel Density Steering for Multi-Task Image Restoration](https://openreview.net/forum?id=JsMABsfR5S) | 2025 | NeurIPS | 多任务图像恢复 | Kernel Density Steering | - |
| [Learning Continuous Wasserstein Barycenter Space for Generalized All-in-One Image Restoration](https://github.com/xl-tang3/BaryIR) | 2026 | TPAMI | 通用 All in one | Wasserstein Barycenter Representation | [code](https://github.com/xl-tang3/BaryIR) |
| [Contrastive Prompt Learning for All-in-One Image Restoration](https://ieeexplore.ieee.org/document/10946393) | 2026 | TPAMI | 通用 All in one | Contrastive Prompt Learning | - |
| [Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration](https://ojs.aaai.org/index.php/AAAI/article/view/37741) | 2026 | AAAI | 夜间多天气恢复 | Multi-weather Nighttime Restoration | - |
| [Gradient as Conditions: Leveraging Degradation Gradients for All-in-One Image Restoration](https://ojs.aaai.org/index.php/AAAI/article/view/37340) | 2026 | AAAI | 通用 All in one | Degradation Gradient Conditioning | - |
| [Human-Visual-Perception-Inspired Image Restoration for Multiple Degradation](https://ojs.aaai.org/index.php/AAAI/article/view/37599) | 2026 | AAAI | 多退化恢复 | Human Visual Perception | - |
| [Test-Time Preference Optimization for Image Restoration](https://ojs.aaai.org/index.php/AAAI/article/view/38154) | 2026 | AAAI | 通用图像恢复 | Test-time Preference Optimization | - |
| [SpikingIR: Spiking Neural Network for Lightweight Image Restoration](https://ojs.aaai.org/index.php/AAAI/article/view/37535) | 2026 | AAAI | 通用图像恢复 | Spiking Neural Network | - |
| [ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Zeng_ShiftLUT_Spatial_Shift_Enhanced_Look-Up_Tables_for_Efficient_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 通用图像恢复 | Look-Up Tables | - |
| [Beyond the Ground Truth: Enhanced Supervision for Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Ryou_Beyond_the_Ground_Truth_Enhanced_Supervision_for_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 通用图像恢复 | Enhanced Supervision | - |
| [NanoSD: Edge Efficient Foundation Model for Real Time Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Sanyal_NanoSD_Edge_Efficient_Foundation_Model_for_Real_Time_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 通用图像恢复基础模型 | Edge-efficient Foundation Model | - |
| [Retrieve-to-Restore: Efficient All-in-One Image Restoration with a Retrieval-Based Degradation Bank](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Retrieve-to-Restore_Efficient_All-in-One_Image_Restoration_with_a_Retrieval-Based_Degradation_Bank_CVPR_2026_paper.html) | 2026 | CVPR | 通用 All in one | Retrieval-based Degradation Bank | [code](https://github.com/cscxwang/R2R) |
| [UniLDiff: Unlocking the Power of Diffusion Priors for All-in-One Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Cheng_UniLDiff_Unlocking_the_Power_of_Diffusion_Priors_for_All-in-One_Image_CVPR_2026_paper.html) | 2026 | CVPR | 通用 All in one | Diffusion Prior | - |
| [CARD: Correlation Aware Restoration with Diffusion](https://openaccess.thecvf.com/content/CVPR2026/html/Nezakati_CARD_Correlation_Aware_Restoration_with_Diffusion_CVPR_2026_paper.html) | 2026 | CVPR | 通用图像恢复 | Correlation-aware Diffusion | - |
| [Degradation-Consistent Test-Time Adaptation for All-in-One Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_Degradation-Consistent_Test-Time_Adaptation_for_All-in-One_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 通用 All in one | Test-Time Adaptation | - |
| [FAPE-IR: Frequency-Aware Planning and Execution Framework for All-in-One Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_FAPE-IR_Frequency-Aware_Planning_and_Execution_Framework_for_All-in-One_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 通用 All in one | MLLM Planning + Diffusion + LoRA-MoE | [code](https://github.com/Programmergg/FAPE-IR) |
| [FoundIR-v2: Optimizing Pre-Training Data Mixtures for Image Restoration Foundation Model](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_FoundIR-v2_Optimizing_Pre-Training_Data_Mixtures_for_Image_Restoration_Foundation_Model_CVPR_2026_paper.html) | 2026 | CVPR | 通用 All in one | Foundation Model + Data Mixture | [code](https://github.com/cschenxiang/FoundIR-v2) |
| [Self-supervised Dynamic Heterogeneous Degradation Modeling for Unified Zero-Shot Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_Self-supervised_Dynamic_Heterogeneous_Degradation_Modeling_for_Unified_Zero-Shot_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 通用 All in one | Zero-shot Degradation Modeling | - |
| [Restore, Assess, Repeat: A Unified Framework for Iterative Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Restore_Assess_Repeat_A_Unified_Framework_for_Iterative_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 通用 All in one | Iterative Restoration + Quality Assessment | [code](https://github.com/SamsungLabs/RAR) |
| [Hybrid Agents for Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Hybrid_Agents_for_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 通用图像恢复 | Hybrid Agents | - |
| [RADAR: VQ-VAE Decoder of VAR is a Good Student for Restoring Against Degradation by Acceleration](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_RADAR_VQ-VAE_Decoder_of_VAR_is_a_Good_Student_for_CVPR_2026_paper.html) | 2026 | CVPR | 通用图像恢复 | VAR/VQ-VAE Distillation | - |
| [UARE: A Unified Vision-Language Model for Image Quality Assessment, Restoration, and Enhancement](https://openaccess.thecvf.com/content/CVPR2026/html/Li_UARE_A_Unified_Vision-Language_Model_for_Image_Quality_Assessment_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 图像质量评价 + 恢复 + 增强 | Vision-Language Model | - |
| [Dynamic Exposure Burst Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Dynamic_Exposure_Burst_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 曝光退化恢复 | Burst Restoration | - |
| [Guiding Diffusion Models with Semantically Degraded Conditions](https://openaccess.thecvf.com/content/CVPR2026/html/Han_Guiding_Diffusion_Models_with_Semantically_Degraded_Conditions_CVPR_2026_paper.html) | 2026 | CVPR | 通用图像恢复 | Semantically Degraded Conditions | - |
| [HalluGen: Synthesizing Realistic and Controllable Hallucinations for Evaluating Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_HalluGen_Synthesizing_Realistic_and_Controllable_Hallucinations_for_Evaluating_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 图像恢复评估 | Hallucination Benchmark | - |
| [Pixel Ignores, Superpixel Sees: Adverse Weather Image Restoration via Semantic-Center SSM](https://github.com/LIDAYU-DayuLi/SSR) | 2026 | ECCV | All in one | Semantic-Center SSM | [code](https://github.com/LIDAYU-DayuLi/SSR) |

### 6.6 通用图像恢复 General Image Restoration

| 论文题目 | 年份 | 会议/期刊| 天气类型 | 方法类别 | 代码/项目链接 |
|---|---:|---|---|---|---|
| [Multi-Stage Progressive Image Restoration](https://openaccess.thecvf.com/content/CVPR2021/html/Zamir_Multi-Stage_Progressive_Image_Restoration_CVPR_2021_paper.html) | 2021 | CVPR | 去雨、去模糊、去噪 | Multi-stage CNN | [code](https://github.com/swz30/MPRNet) |
| [Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html) | 2022 | CVPR | 去雨、去模糊、去噪等 | Transformer | [code](https://github.com/swz30/Restormer) |
| [MAXIM: Multi-Axis MLP for Image Processing](https://openaccess.thecvf.com/content/CVPR2022/html/Tu_MAXIM_Multi-Axis_MLP_for_Image_Processing_CVPR_2022_paper.html) | 2022 | CVPR | 通用图像恢复 | MLP | [code](https://github.com/google-research/maxim) |
| [Uformer: A General U-Shaped Transformer for Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Uformer_A_General_U-Shaped_Transformer_for_Image_Restoration_CVPR_2022_paper.html) | 2022 | CVPR | 通用图像恢复 | Transformer | [code](https://github.com/ZhendongWang6/Uformer) |
| [Simple Baselines for Image Restoration](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3043_ECCV_2022_paper.php) | 2022 | ECCV | 去雨、去模糊、去噪等 | CNN | [code](https://github.com/megvii-research/NAFNet) |
| [MambaIR: A Simple Baseline for Image Restoration with State-Space Model](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6810_ECCV_2024_paper.php) | 2024 | ECCV | 通用图像恢复 | State Space Model | [code](https://github.com/csguoh/MambaIR) |
| [Degradation-Aware Residual-Conditioned Optimal Transport for Unified Image Restoration](https://ieeexplore.ieee.org/document/10970105/) | 2025 | TPAMI | 通用图像恢复 | Optimal Transport | - |
| [MB-TaylorFormer V2: Improved Multi-Branch Linear Transformer Expanded by Taylor Formula for Image Restoration](https://ieeexplore.ieee.org/document/10962317/) | 2025 | TPAMI | 去雾、去雨、去雪、去模糊等 | Multi-Branch Linear Transformer | - |
| [Dual Prompting Image Restoration with Diffusion Transformers](https://openaccess.thecvf.com/content/CVPR2025/html/Kong_Dual_Prompting_Image_Restoration_with_Diffusion_Transformers_CVPR_2025_paper.html) | 2025 | CVPR | 通用图像恢复 | Diffusion Transformer | - |
| [A Regularization-Guided Equivariant Approach for Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Bai_A_Regularization-Guided_Equivariant_Approach_for_Image_Restoration_CVPR_2025_paper.html) | 2025 | CVPR | 通用图像恢复 | Equivariant Regularization | - |
| [Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](https://openaccess.thecvf.com/content/CVPR2025/html/Garber_Zero-Shot_Image_Restoration_Using_Few-Step_Guidance_of_Consistency_Models_and_CVPR_2025_paper.html) | 2025 | CVPR | 通用图像恢复 | Consistency Model Guidance | - |
| [A Universal Scale-Adaptive Deformable Transformer for Image Restoration across Diverse Artifacts](https://openaccess.thecvf.com/content/CVPR2025/html/He_A_Universal_Scale-Adaptive_Deformable_Transformer_for_Image_Restoration_across_Diverse_CVPR_2025_paper.html) | 2025 | CVPR | 通用图像恢复 | Scale-Adaptive Deformable Transformer | - |
| [FiRe: Fixed-points of Restoration Priors for Solving Inverse Problems](https://openaccess.thecvf.com/content/CVPR2025/html/Terris_FiRe_Fixed-points_of_Restoration_Priors_for_Solving_Inverse_Problems_CVPR_2025_paper.html) | 2025 | CVPR | 通用逆问题/图像恢复 | Restoration Priors | - |
| [From Zero to Detail: Deconstructing Ultra-High-Definition Image Restoration from Progressive Spectral Perspective](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_From_Zero_to_Detail_Deconstructing_Ultra-High-Definition_Image_Restoration_from_Progressive_CVPR_2025_paper.html) | 2025 | CVPR | UHD 图像恢复 | Progressive Spectral Modeling | - |
| [Navigating Image Restoration with VAR's Distribution Alignment Prior](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Navigating_Image_Restoration_with_VARs_Distribution_Alignment_Prior_CVPR_2025_paper.html) | 2025 | CVPR | 通用图像恢复 | VAR Distribution Alignment Prior | - |
| [MambaIRv2: Attentive State Space Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html) | 2025 | CVPR | 通用图像恢复 | State Space Model | [code](https://github.com/csguoh/MambaIR) |
| [MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Li_MaIR_A_Locality-_and_Continuity-Preserving_Mamba_for_Image_Restoration_CVPR_2025_paper.html) | 2025 | CVPR | 通用图像恢复 | State Space Model | [code](https://github.com/XLearning-SCU/2025-CVPR-MaIR) |
| [ACL: Activating Capability of Linear Attention for Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Gu_ACL_Activating_Capability_of_Linear_Attention_for_Image_Restoration_CVPR_2025_paper.html) | 2025 | CVPR | 通用图像恢复 | Linear Attention | - |
| [Reversing Flow for Image Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Qin_Reversing_Flow_for_Image_Restoration_CVPR_2025_paper.html) | 2025 | CVPR | 通用图像恢复 | Normalizing Flow | - |
| [FoundIR: Unleashing Million-scale Training Data to Advance Foundation Models for Image Restoration](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_FoundIR_Unleashing_Million-scale_Training_Data_to_Advance_Foundation_Models_for_Image_ICCV_2025_paper.html) | 2025 | ICCV | 通用图像恢复 | Foundation Model | [code](https://github.com/House-Leo/FoundIR) |
| [Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](https://openaccess.thecvf.com/content/ICCV2025/html/Hu_Enhancing_Image_Restoration_Transformer_via_Adaptive_Translation_Equivariance_ICCV_2025_paper.html) | 2025 | ICCV | 通用图像恢复 | Translation Equivariance | - |
| [Exploiting Diffusion Prior for Task-driven Image Restoration](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Exploiting_Diffusion_Prior_for_Task-driven_Image_Restoration_ICCV_2025_paper.html) | 2025 | ICCV | 任务驱动图像恢复 | Diffusion Prior | - |
| [EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](https://openaccess.thecvf.com/content/ICCV2025/html/Lin_EAMamba_Efficient_All-Around_Vision_State_Space_Model_for_Image_Restoration_ICCV_2025_paper.html) | 2025 | ICCV | 通用图像恢复 | State Space Model | - |
| [Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_Devil_is_in_the_Uniformity_Exploring_Diverse_Learners_within_Transformer_ICCV_2025_paper.html) | 2025 | ICCV | 通用图像恢复 | Diverse Transformer Learners | - |
| [Frequency-Guided Posterior Sampling for Diffusion-Based Image Restoration](https://openaccess.thecvf.com/content/ICCV2025/html/Thaker_Frequency-Guided_Posterior_Sampling_for_Diffusion-Based_Image_Restoration_ICCV_2025_paper.html) | 2025 | ICCV | 通用图像恢复 | Frequency-guided Diffusion Sampling | - |
| [Reverse Convolution and Its Applications to Image Restoration](https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Reverse_Convolution_and_Its_Applications_to_Image_Restoration_ICCV_2025_paper.html) | 2025 | ICCV | 通用图像恢复 | Reverse Convolution | - |
| [Residual Diffusion Bridge Model for Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Residual_Diffusion_Bridge_Model_for_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 通用图像恢复 | Diffusion Bridge | [code](https://github.com/MiliLab/RDBM) |
| [Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Xiao_Beyond_Ground-Truth_Leveraging_Image_Quality_Priors_for_Real-World_Image_Restoration_CVPR_2026_paper.html) | 2026 | CVPR | 真实图像恢复 | Image Quality Priors | - |
| [Scan Clusters, Not Pixels: A Cluster-Centric Paradigm for Efficient Ultra-high-definition Image Restoration](https://openaccess.thecvf.com/content/CVPR2026/html/Wu_Scan_Clusters_Not_Pixels_A_Cluster-Centric_Paradigm_for_Efficient_Ultra-high-definition_CVPR_2026_paper.html) | 2026 | CVPR | UHD 去雨、去雾、低照度、去雪 | Cluster-Centric SSM | [code](https://github.com/5chen/C2SSM) |

### 6.7 视频不利天气/视频恢复 Video Restoration

| 论文题目 | 年份 | 会议/期刊| 天气类型 | 方法类别 | 代码/项目链接 |
|---|---:|---|---|---|---|
| [Video Adverse-Weather-Component Suppression Network via Weather Messenger and Adversarial Backpropagation](https://doi.org/10.1109/ICCV51070.2023.01214) | 2023 | ICCV | 视频 All in one | CNN + Transformer | [code](https://github.com/scott-yjyang/ViWS-Net) |
| [Video Dehazing via a Multi-Range Temporal Alignment Network with Physical Prior](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_Video_Dehazing_via_a_Multi-Range_Temporal_Alignment_Network_With_Physical_CVPR_2023_paper.html) | 2023 | CVPR | 视频去雾 | Temporal Alignment + Physical Prior | [code](https://github.com/jiaqixuac/MAP-Net) |
| [All-in-one Video Restoration for Time-varying Unknown Degradations](https://openreview.net/forum?id=9G7pACD1jD) | 2024 | NeurIPS | 通用视频恢复 | Video Restoration | [code](https://github.com/XLearning-SCU/2024-NeurIPS-AverNet) |
| [SeedVR: Seeding Infinity in Diffusion Transformer Towards Generic Video Restoration](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_SeedVR_Seeding_Infinity_in_Diffusion_Transformer_Towards_Generic_Video_Restoration_CVPR_2025_paper.html) | 2025 | CVPR | 通用视频恢复 | Diffusion Transformer | - |
| [Event-guided Unified Framework for Low-light Video Enhancement, Frame Interpolation, and Deblurring](https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Event-guided_Unified_Framework_for_Low-light_Video_Enhancement_Frame_Interpolation_and_ICCV_2025_paper.html) | 2025 | ICCV | 低照度视频增强 | Event-guided Unified Framework | - |
| [VSRELL: A Simple Baseline for Video Super-Resolution and Enhancement in Low-Light Environment](https://openaccess.thecvf.com/content/CVPR2026/html/Hui_VSRELL_A_Simple_Baseline_for_Video_Super-Resolution_and_Enhancement_in_CVPR_2026_paper.html) | 2026 | CVPR | 低照度视频增强 | Video SR + Enhancement | - |


## 7. 数据集

> 聚焦不利天气图像/视频恢复。

### 7.1 去雨与雨滴去除 Deraining / Raindrop Removal

| 数据集名称 | 年份 | 任务 | 天气类型 | 数据类型 | 是否配对 | 图像数量 | 分辨率 | 训练/测试划分 | 合成/真实 | 标注类型 | 下载链接 |
|---|---:|---|---|---|---|---:|---|---|---|---|---|
| DDN-Data | 2017 | 去雨 | 雨线 | 图像 | 是 | 14K | 未说明 | 训练/测试划分见官方项目 | 合成 | clean image | [Project](https://xueyangfu.github.io/projects/cvpr2017.html) |
| Rain800 | 2017 | 去雨 | 雨线 | 图像 | 是 | 800 | 未说明 | 700 train / 100 test | 合成 | clean image | [GitHub](https://github.com/guanqiyuan/Rain800) |
| Rain200L/H | 2017 | 去雨 | 小雨/大雨 | 图像 | 是 | 4K | 未说明 | Rain200L 与 Rain200H 各 1,800 train / 200 test | 合成 | clean image | [Official](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html) |
| DID-Data / Rain1200 | 2018 | 去雨 | 雨线 | 图像 | 是 | 13.2K | 未说明 | 12,000 train / 1,200 test | 合成 | clean image、rain density | [DID-MDN](https://github.com/hezhangsprinter/DID-MDN) / [GCANet](https://github.com/wfs123456/GCANet) |
| RainDrop / Raindrop-A / TestA | 2018 | 雨滴去除 | 雨滴 | 图像 | 是 | 约 1.1K | 未说明 | All-weather 协议常用 818 train / 58 test | 真实雨滴 + 配对 GT | clean image | [Project](https://rui1996.github.io/raindrop/raindrop_removal.html) / [GitHub](https://github.com/rui1996/DeRaindrop) |
| SPA-Data | 2019 | 真实去雨 | 雨 | 图像 | 是 | 约 29.5K 或更大规模版本 | 未说明 | CyclicPrompt、AllRestorer 用于真实雨天测试 | 真实 | clean image | [SPANet](https://github.com/stevewongv/SPANet) |
| RainDS / RainDS-real | 2021 | 真实去雨、雨滴/雨线恢复 | 雨、雨滴 | 图像 | 是/部分无 GT | 5.8K | 未说明 | 含 rain streak only、raindrop only、混合雨退化；部分真实测试无 GT | 合成 + 真实 | clean image / 无 GT | [GitHub](https://github.com/Songforrr/RainDS_CCN) |
| RainMotion | 2022 | 视频去雨 | 雨 | 视频 | 是 | 80 个视频，2,800 帧 | 未说明 | 40 train / 40 test；2,000 train frames / 800 test frames | 合成 | clean video frames | [RDD-Net](https://github.com/wangshauitj/RDD-Net) |
| RealRain-1k | 2022 | 真实去雨 | 雨线、雨累积 | 图像 | 是 | 1,120 对 | 高分辨率，具体未说明 | 低密度/高密度雨纹子集 | 真实视频生成配对 | clean image、rain streak layer | [GitHub](https://github.com/hiker-lw/RealRain-1k) |
| GTAV-NightRain | 2022 | 夜间去雨 | 夜间雨线 | 图像 | 是 | 12,860 rainy / 1,286 clean | HD | set1 / set2 / set3 | GTA V 合成 | clean image | [GitHub](https://github.com/zkawfanx/GTAV-NightRain) |
| GT-RAIN | 2022 | 真实去雨 | 雨线、雨累积 | 图像/视频帧 | 是 | 31,524 对 | 未说明 | 26,124 train / 3,300 val / 2,100 test | 真实 | clean image | [Project](https://visual.ee.ucla.edu/gt_rain.htm/) / [GitHub](https://github.com/UCLA-VMG/GT-RAIN) |
| LWDDS | 2023 | 视频雨滴去除 | 雨滴 | 视频 | 是 | 68,100 对帧 | 未说明 | 45 train clips / 6 test clips；67,500 train / 600 test | 合成 | clean video frames | [GitHub](https://github.com/csqiangwen/Video_Waterdrop_Removal_in_Driving_Scenes) |
| VRDS | 2023 | 视频去雨滴 + 去雨线 | 雨滴、雨线 | 视频 | 是 | 102 个视频 | 1280 x 720 | 72 train / 30 test | 合成 | clean video frames、rain mask | [GitHub](https://github.com/TonyHongtaoWu/ViMP-Net) |
| LHP-Rain | 2023 | 真实去雨 | 雨线、雨雾、地面飞溅 | 视频帧/图像 | 是 | 3,000 视频序列，约 1M 帧对 | 1920 x 1080 | 官方 train/test | 真实 | clean frame | [GitHub](https://github.com/yunguo224/LHP-Rain) |
| HQ-RAIN | 2023 | 去雨统一评估 | 雨线 | 图像 | 是 | 5,000 对 | 高分辨率，具体未说明 | 官方 train/test | 合成 | clean image | [GitHub](https://github.com/cschenxiang/HQ-RAIN) |
| UAV-Rain1k | 2024 | 无人机雨滴去除 | 雨滴 | 图像 | 是 | 1,020 对 | 平均约 1500 x 1000 | 800 train / 220 test | 合成 | clean image、雨滴密度标签 | [GitHub](https://github.com/cschenxiang/UAV-Rain1k) |
| RoadScene-rain | 2024 | 夜间去雨 | 夜间雨天驾驶 | 图像 | 是 | 221 对 | 未说明 | 181 train / 40 test | 合成 | clean image / RGB-IR reference | [GitHub](https://github.com/CidanShi/NiteDR-Nighttime-Image-De-raining) |
| Raindrop Clarity | 2024 | 雨滴去除 | 白天/夜间雨滴 | 图像 | 是 | 15,186 对/三元组 | 未说明 | 白天 5,442；夜间 9,744 | 真实 | clean background、focus variants | [GitHub](https://github.com/jinyeying/RaindropClarity) |
| HQ-NightRain | 2025 | 夜间去雨 | 夜间雨线、雨滴、混合雨 | 图像 | 是 | 10,000 train / 900 val / 300 test；另含 512 真实夜雨图像 | 1280 x 720 | 10,000 train / 900 val / 300 test | 合成 + 真实 | clean image、real rainy subset | [GitHub](https://github.com/guanqiyuan/CST-Net) |

### 7.2 去雾与去烟 Dehazing / Desmoking

| 数据集名称 | 年份 | 任务 | 天气类型 | 数据类型 | 是否配对 | 图像数量 | 分辨率 | 训练/测试划分 | 合成/真实 | 标注类型 | 下载链接 |
|---|---:|---|---|---|---|---:|---|---|---|---|---|
| WaterlooIVC Dehazed Image Database | 2015 | 去雾主观评价 | 雾/霾 | 图像 | 否 | 25 组 | 未说明 | 25 张真实有雾图像，每张对应 8 种算法结果 | 真实 | human perceptual score / dehazed outputs | [Official](https://ivc.uwaterloo.ca/database/dehaze.html) |
| D-HAZY | 2016 | 去雾 | 雾/霾 | 图像 | 是 | 约 1.4K | 640 x 480 等 | 基于 NYU-Depth v2 和 Middlebury 合成 | 合成 | clean image、depth | [Baidu](https://pan.baidu.com/s/1r_ByUn1wvbBgMBxbmTOF2Q) |
| O-HAZE | 2018 | 去雾 | 室外雾 | 图像 | 是 | 45 | 未说明 | NTIRE 2018 Outdoor Dehazing | 真实人工造雾 | clean image | [Official](https://data.vision.ee.ethz.ch/cvl/ntire18//o-haze/) |
| I-HAZE | 2018 | 去雾 | 室内雾 | 图像 | 是 | 35 | 未说明 | NTIRE 2018 Indoor Dehazing | 真实人工造雾 | clean image | [Official](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/) |
| RESIDE / SOTS-outdoor | 2019 | 去雾 | 雾/霾 | 图像 | 是 | 约 87K | 未说明 | 含 ITS、OTS、SOTS、RTTS、HSTS；ARHC 使用 SOTS-outdoor 评估 | 合成 + 真实 | clean image | [Official](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) |
| Dense-Haze | 2019 | 去雾 | 浓雾 | 图像 | 是 | 55 | 未说明 | NTIRE 2019 Dense Haze | 真实人工造雾 | clean image | [Official](https://data.vision.ee.ethz.ch/cvl/ntire19//dense-haze/) |
| NH-HAZE | 2020 | 去雾 | 非均匀雾/霾 | 图像 | 是 | 55 | 未说明 | NTIRE 2020 challenge 数据 | 真实 | clean image | [Official](https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/) |
| REVIDE | 2021 | 视频去雾 | 雾 | 视频 | 是 | 48 个视频，1,082 帧 | 未说明 | 42 train / 6 test；928 train frames / 154 test frames | 真实 | clean video frames | [GitHub](https://github.com/BookerDeWitt/REVIDE_Dataset) |
| Haze4K | 2021 | 去雾 | 雾/霾 | 图像 | 是 | 4K | 未说明 | 3,000 train / 1,000 test | 合成 | clean image、transmission、大气光 | [DMT-Net](https://github.com/liuye123321/DMT-Net) |
| RIDCP Synthetic Dehazing Data | 2023 | 去雾 | 真实感雾、低光、噪声、JPEG 压缩 | 图像 | 是 | 由退化管线生成 | 未说明 | 训练数据由官方管线生成 | 合成 | clean image | [GitHub](https://github.com/RQ-Wu/RIDCP_dehazing) |
| SmokeBench | 2025 | 去烟雾 | 火灾烟雾/监控烟雾 | 图像 | 是 | 具体数量见仓库 | 未说明 | 官方 train/test | 真实采集 | clean image / smoke image | [GitHub](https://github.com/ncfjd/SmokeBench) |

### 7.3 去雪 Desnowing

| 数据集名称 | 年份 | 任务 | 天气类型 | 数据类型 | 是否配对 | 图像数量 | 分辨率 | 训练/测试划分 | 合成/真实 | 标注类型 | 下载链接 |
|---|---:|---|---|---|---|---:|---|---|---|---|---|
| Snow100K | 2018 | 去雪 | 雪 | 图像 | 是 | 100K | 最大边长约 640 | 50K train / 50K test；常用 Snow100K-S/M/L 子测试集 | 合成 | clean image、snow mask | [Official](https://sites.google.com/view/yunfuliu/desnownet) |
| Snow100K-real / Snow100K-R | 2018 | 真实去雪评估 | 雪 | 图像 | 否 | 1,329 | 最大边长约 640 | 仅真实测试/定性评估 | 真实 | 无 GT | [Official](https://sites.google.com/view/yunfuliu/desnownet) |
| CSD | 2021 | 去雪 | 雪、雪雾/veil | 图像 | 是 | 10K | 未说明 | MWFormer-real 将其训练集加入训练 | 合成 | clean image | [GitHub](https://github.com/weitingchen83/ICCV2021-Single-Image-Desnowing-HDCWNet) |
| KITTI-snow | 2023 | 视频去雪 | 雪 | 视频 | 是 | 50 个视频，2,500 帧 | 未说明 | 35 train / 15 test；1,750 train frames / 750 test frames | 合成 | clean video frames | [ViWS-Net](https://github.com/scott-yjyang/ViWS-Net) |

### 7.4 低照度与夜间增强 Low-Light / Nighttime Enhancement

| 数据集名称 | 年份 | 任务 | 天气类型 | 数据类型 | 是否配对 | 图像数量 | 分辨率 | 训练/测试划分 | 合成/真实 | 标注类型 | 下载链接 |
|---|---:|---|---|---|---|---:|---|---|---|---|---|
| SICE | 2018 | 低照度增强 | 夜间/低照度 | 图像 | 是 | 589 组序列，4,413 张图像 | 高分辨率，具体未说明 | Part1: 360 sequences；Part2: 229 sequences | 真实/多曝光 | reference image | [GitHub](https://github.com/csjcai/SICE) |
| LOL-v1 | 2018 | 低照度增强 | 低照度/夜间 | 图像 | 是 | 500 | 400 x 600 | 485 train / 15 test | 真实 + 合成 | normal-light reference | [Project](https://daooshee.github.io/BMVC2018website/) |
| SID | 2018 | 极低照度增强 | 夜间/极低照度 | RAW 图像 | 是 | 5,094 | RAW 原始分辨率 | Sony / Fuji 子集，短曝光输入与长曝光参考 | 真实 | long-exposure reference | [Project](https://cchen156.github.io/SID.html) / [GitHub](https://github.com/cchen156/Learning-to-See-in-the-Dark) |
| LOL-v2 | 2020 | 低照度增强 | 低照度/夜间 | 图像 | 是 | Real: 789；Synthetic: 1,000 | 未说明 | Real: 689 train / 100 test；Synthetic: 900 train / 100 test | 真实 + 合成 | clean image | [CVPR-2020-Semi-Low-Light](https://github.com/flyywh/CVPR-2020-Semi-Low-Light/) |
| SDSD | 2021 | 低照度视频增强 | 夜间/低照度 | 视频 | 是 | 150 对视频，37,500 帧 | 1920 x 1080 | 70 indoor / 80 outdoor | 真实 | normal-light video frames | [GitHub](https://github.com/dvlab-research/SDSD) |
| LSRW | 2021 | 低照度增强 | 低照度/夜间 | 图像 | 是 | 5,650 对 | 未说明 | 官方 train/test | 真实 | normal-light reference | [GitHub](https://github.com/JianghaiSCU/R2RNet) |
| DID | 2023 | 低照度视频增强 | 低照度/夜间 | 视频 | 是 | 413 对视频，41,038 帧 | 560 x 1440 / 最高 2560 x 1440 | 官方 train/test | 真实 | normal-light video frames | [GitHub](https://github.com/ciki000/DID) |
| UHD-LOL | 2023 | 超高清低照度增强 | 低照度/夜间 | 图像 | 是 | 约 11K 对 | 4K / 8K | UHD-LOL4K 与 UHD-LOL8K train/test | 合成 | normal-light reference | [LLFormer](https://github.com/TaoWangzj/LLFormer) |
| UHD-LL | 2023 | 超高清低照度增强 | 低照度/夜间 | 图像 | 是 | 2,150 对 | 4K UHD | 2,000 train / 150 test | 真实 | normal-light reference | [UHDFour](https://li-chongyi.github.io/UHDFour/) / [GitHub](https://github.com/Li-Chongyi/UHDFour) |
| RELED | 2024 | 低照度视频增强 + 去模糊 | 低照度/夜间 | 视频 + event | 是 | 42 个城市场景 | 1024 x 768 | 官方 train/test | 真实 | normal-light sharp image、event stream | [GitHub](https://github.com/intelpro/ELEDNet) |

### 7.5 All-in-One / 多天气与复合退化

| 数据集名称 | 年份 | 任务 | 天气类型 | 数据类型 | 是否配对 | 图像数量 | 分辨率 | 训练/测试划分 | 合成/真实 | 标注类型 | 下载链接 |
|---|---:|---|---|---|---|---:|---|---|---|---|---|
| Outdoor-Rain / Test1 | 2019 | 去雨 + 去雾 | 雨线、雾/霾 | 图像 | 是 | 约 10.5K | 高分辨率，具体未说明 | 常用 8,250 train / 750 test | 合成 | clean image | [TransWeather](https://github.com/jeya-maria-jose/TransWeather) / [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion) |
| All-weather / AllWeather | 2022 | 多天气统一恢复 | 雨雾、雪、雨滴 | 图像 | 是 | 18,069 train；测试集约 16K-17K | 未说明 | 训练集由 Snow100K、Outdoor-Rain、RainDrop 组成；测试包含 Test1、RainDrop TestA、Snow100K-L/S | 合成 + 真实雨滴 | clean image | [TransWeather](https://github.com/jeya-maria-jose/TransWeather) |
| WeatherStream | 2023 | 多天气统一恢复 | 雨、雪、雾 | 视频帧/图像 | 是 | 187,500 | 未说明 | 176,100 train / 11,400 test | 真实/自动采集 | clean image | [WGWS-Net](https://github.com/zhuyr97/WGWS-Net) |
| CDD-11 | 2024 | 多天气统一恢复、复合退化恢复 | 低照度、雾、雨、雪及其复合退化 | 图像 | 是 | 11 类退化，具体数量见仓库 | 1080 x 720 原图；训练 patch 256 x 256 | 官方 train/test | 合成 + 真实 | clean image | [HuggingFace](https://huggingface.co/datasets/gy65896/CDD-11) / [OneRestore](https://github.com/gy65896/OneRestore) |
| HAC | 2025 | 任意混合不利天气恢复 | 雾、雨、雪、夜间、雨滴 | 图像 | 是 | 约 316K | 未说明 | 含 31 种退化组合；训练集自动生成，测试集人工调制 | 合成 | clean image、退化类型标签 | [GitHub](https://github.com/arun-kollan/RAHC) / [Paper](https://arxiv.org/abs/2305.09996) |
| WeatherBench | 2025 | 真实多天气统一恢复 | 雨、雪、雾、昼夜光照 | 图像 | 是 | 42,002 对 | 512 x 512 | 41,402 train / 600 test | 真实 | clean image | [GitHub](https://github.com/guanqiyuan/WeatherBench) |

### 7.6 下游夜间感知评估 Downstream Nighttime Evaluation

| 数据集名称 | 年份 | 任务 | 天气类型 | 数据类型 | 是否配对 | 图像数量 | 分辨率 | 训练/测试划分 | 合成/真实 | 标注类型 | 下载链接 |
|---|---:|---|---|---|---|---:|---|---|---|---|---|
| Dark Zurich | 2019 | 下游语义分割评估 | 夜间/低照度 | 图像 | 否/部分配对 | 8,779 | 未说明 | 含 nighttime、dusk、daytime；用于夜间语义分割 | 真实 | semantic segmentation labels | [MGCDA](https://github.com/sakaridis/MGCDA) |


## 9. 评价指标

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

