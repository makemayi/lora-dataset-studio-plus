# MOSS-Image 人物 LoRA 训练指南（LDS + ai-toolkit，3090 24G）

## 1. 固定设置（不要改）

| 项 | 值 | 说明 |
|---|---|---|
| 底模 | MOSS-Image-Base（或本地 convrot int8 三件套） | OpenMOSS 官方推荐用 Base 训 LoRA，成品可用于 SFT |
| 量化 | DiT + TE 都开，`convrot8` | 需 triton-windows（已装），否则很慢 |
| Text Encoder | 冻结（`train_text_encoder: false`） | 官方硬要求 |
| batch / 梯度累积 | 1 / 1 | |
| 梯度检查点 | **开** | 关了 768/1024 桶会溢出显存，Windows 不报 OOM，而是换页卡死（GPU 100%、步数不动） |
| 第 22 层 | **LoRA 不训** | LDS 自动写入，见第 2 节 |
| 网络 | LoRA rank 32 / alpha 32 | |
| 优化器 | adamw8bit | |
| 分辨率 | 768 + 1024 | 人脸细节要 1024，不要只用 512 |
| caption | 同名 `.txt`，触发词统一（如 `JCL001`），描述画面真实内容 | 发型发色写进 caption，否则会被吸进触发词 |
| caption dropout | 0.05 | |
| 噪声调度 | flowmatch | |

## 2. 第 22 层不训（自动）

LDS 生成的 MOSS 任务配置自动带两项（与 ai-toolkit 官方 MOSS 预设一致）：

```yaml
model:
  model_kwargs:
    uncond_skip_layers: [22]        # CFG 负分支跳过第 22 块（0 起算）
network:
  network_kwargs:
    ignore_if_contains: ["layers.22."]   # LoRA 不挂第 22 块
```

原因：推理时 CFG 负分支跳过第 22 块（官方 SLG 默认）。LoRA 不碰这一块，CFG 跳块对比就始终是原模型的，画质不被 LoRA 带偏。两项必须成对，不要单独删。

注意：
- 改动后需重启 LDS 后端才生效（重启前确认没有任务在跑）。
- 旧 LoRA 续训：新配置不再训第 22 层，旧权重里第 22 层部分会被忽略。建议新开一轮训练，不要续旧的。
- 检查方法：训练目录里的 `config.yaml` 应能看到上面两项。

## 3. Timestep 选 `moss`

训练面板 → Timestep type → 选 **moss**（只有 MOSS 家族显示这一项）。

- 含义：MOSS 预训练用的 logit-normal 噪声分布（mu = -1.09，s = 0.95，截断 0.95，另加 1% 均匀采样），中位 sigma ≈ 0.78，偏高噪声段，对应构图/身份。
- 依赖本地 ai-toolkit 补丁（`toolkit/samplers/custom_flowmatch_sampler.py`）。升级或重装 ai-toolkit 后要确认补丁还在，否则会报未知 timestep。
- 对照：ai-toolkit 官方预设是 `shift`，旧版 LDS 默认是 `sigmoid`。效果有疑问时用同一数据集、同一种子做 A/B，用 ArcFace 相似度判断。

## 4. 按图片数设步数：50 或 60 轮

batch = 1 时，1 轮 = 每张图看一次：

```
总步数 = 图片数 × 轮数
```

选择规则：
- **图片 ≤ 30 张：60 轮**（图少，每张多看几遍才能锁住身份）
- **图片 > 30 张：50 轮**（图多，角度覆盖足，少看几遍防过拟合）
- 结果向上取整到 250 的倍数（与保存间隔对齐，最后一个 checkpoint 正好落在终点）

| 图片数 | 轮数 | 计算 | 设置步数 |
|---|---|---|---|
| 15 | 60 | 900 | 1000 |
| 20 | 60 | 1200 | 1250 |
| 25 | 60 | 1500 | 1500 |
| 30 | 60 | 1800 | 2000 |
| 40 | 50 | 2000 | 2000 |
| 50 | 50 | 2500 | 2500 |
| 60 | 50 | 3000 | 3000 |

- LDS 对 MOSS 人物数据集的「推荐步数」已按上表自动计算（开了梯度累积会再除以累积数）。套用了带「每张步数」的预设时，以预设为准。
- 超过 3000 步通常无收益，图很多时宁可删重复角度。
- 数据集 51（40 张）：50 轮 = **2000 步**。

## 5. 学习率

- 默认 **2e-4**（本地 jcl001 实测跑通）。
- 后期出现以下情况，下一轮降到 **1e-4**：角色变僵、姿势/背景固定、没写触发词也出现这张脸。
- 一次只改一个变量（学习率 / 步数 / timestep 分开测）。

## 6. 存档与挑选

- 每 250 步存档 + 出预览，保留 10 个 checkpoint（人物预设默认）。
- 预览：CFG 3，空负面，832×1216。
- 挑选：最终候选用 **50 步** 重出图再比较（官方推荐推理 ≥ 50 步，25/30 步只用于监控）。
- 用 LDS 的最佳 epoch 评分（InsightFace / ArcFace）辅助判断：选**最早**能稳住相似度、换场景仍听提示词的那个，不一定是最后一个。
- 测试提示词四类：贴近训练集 / 换角度换场景 / 脸部细节特写 / 不带触发词（检查是否污染）。

## 7. 速度与显存

- 3090 上 batch 1、梯度检查点开：显存约 20G 内，具体每步耗时以实测日志为准。
- 训练前关掉 ComfyUI / ollama 等占显存的程序。
- 训练中不要重启 LDS 后端（任务队列在内存里，重启会丢任务）。

## 8. 开训前检查清单

- [ ] 图片 + 同名 `.txt` caption，触发词一致
- [ ] Timestep = `moss`
- [ ] 步数 = 图片数 × 50/60，取整到 250
- [ ] rank 32 / alpha 32，lr 2e-4，分辨率 768+1024
- [ ] 梯度检查点开，batch 1
- [ ] LDS 后端已重启到含第 22 层排除的版本（看 `config.yaml`）
- [ ] 显存已清空

## 9. 另一条路：Fizgig 后端

训练面板 → 家族选 MOSS-Image → 训练器下拉选 **Fizgig**（需先在 设置 ▸ Local tools ▸ Fizgig 填目录，如 `E:\Fizgig`）。

跑的是 Fizgig 的 MOSS-Image **Standard** 预设（joyce001 实测：40 张，2800 步，约 3.1 s/步，显存峰值 10.5 GB）：

| 项 | 值 |
|---|---|
| 底模 / 预览 | Base int8 convrot 训，SFT int8 convrot 出预览 |
| LoRA | rank 32 / alpha 32，**第 22 层也训**（Fizgig 实测排除会损失相似度） |
| 优化 | AdamW，固定 lr 1e-4，EMA 0.98 |
| 轮数 | 70（面板 Epochs 可改），batch 1，992² 分桶 |
| 噪声 | clean-end 30%（`lownoise_pct=30`） |
| 存档 / 预览 | 每 5 轮；预览 res_multistep 30 步，CFG 4 |

- 面板只有 rank / alpha / 学习率 / 轮数会生效，其余都由预设固定。步数输入框在这条路上不起作用。
- 模型从 ComfyUI models 目录找：`diffusion_models/moss_image_base_int8_convrot`、`clip/qwen3.5_9b_int8_convrot`、`vae/flux2-vae_small`。
- 每次启动都从零训，旧的 run 目录会被重命名归档（`*_archived_<时间>`），不会删除。
- 跑完后，最终 LoRA 自动部署到 `loras/moss/`。中间存档在 `<Fizgig>/output_loras/lds/u<用户>_<触发词>/output/`。
- 暂不支持：排队、▶ Continue、在 LDS 存档列表里挑中间存档。
- 在 Base 上训，不加 `lora_sft=1`（那是在 SFT 上训练时用的选项，只训注意力）。

## 资料来源

- OpenMOSS USERGUIDE：https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md
- ai-toolkit MOSS 预设：`F:\ai-toolkit\extensions_built_in\diffusion_models\ui.tsx`
- RunComfy MOSS-Image Base 训练指南：https://www.runcomfy.com/trainer/ai-toolkit/moss-image-base-lora-training
