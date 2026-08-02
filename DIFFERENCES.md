# 本分支 vs 上游 perfectgf/lora-dataset-studio

本分支（`makemayi/lora-dataset-studio-plus`）已经完整合并了上游 `main` 的全部历史，
再叠加了以下上游没有的功能和修复。上游仍在开发，本分支不再跟进同步，独立维护。

## 🎲 Quick generate（一键批量生成，全新功能）

不用一张一张挑卡片：设一个总数 + 脸/半身/全身比例 + 每种框选的角度比例，点一下，
从独立的组件库（角度/表情/姿势/服装/背景）里随机组一批不重复的提示词直接提交生成。

- NSFW 比例滑块，只在本地引擎（Klein/Krea）生效，绝不进 API 引擎
- 组件库可通过 config 扩展，不用改代码
- 用户可自定义追加组件（JSON 编辑器）
- 各种边界情况修得比较细：负数比例、重复提交、姿势跟角度冲突（比如"坐姿"卡片混进"站姿"角度池）、脸部提示词照抄参考图背景/服装等

## 🎭 单张换脸（Face swap）

每张卡片一个 🎭↔ 按钮，用固定的 Klein 换脸工作流，把这张图的脸换成数据集参考照片的脸，原地覆盖，不用重新生成整张图。

## Krea 2 Edit 增强

- **角色 LoRA 链**：除了身份 LoRA，额外挂最多 5 个你自己训练的角色 LoRA，逐个可调强度
- **两阶段采样器**：可选的低分辨率预采样 + 放大二阶段，含独立的 step/handoff/分辨率设置
- **per-framing 精调**：reference grounding、reference boost 都可以按脸/半身/全身/背影分别设置，不用一个滑块套所有框选
- 角色 LoRA 路径分隔符自动归一化（Windows/Linux 兼容，正斜杠自动转成 ComfyUI 期望的系统分隔符）

## OneTrainer 支持（本地训练第二后端）

除了 ai-toolkit，现在 Krea 2 数据集可以选 OneTrainer 训练：
- 训练面板加了 ai-toolkit / OneTrainer 选择器
- Settings 里配置 OneTrainer 安装路径
- 支持 OFT_2 微调方式作为 LoRA 之外的选项
- rank/alpha/batch_size 联动修正（原来改 rank 不带 alpha 会导致 LoRA 强度不足的静默 bug）

## Qwen Image（阿里 DashScope）API 引擎

新的变化生成引擎选项，含完整的模型名/尺寸格式/响应解析修复（上游没有这个引擎）。

## 变化目录（Variation catalog）多轮扩充

- 服装色板：从最初的十几件扩到 71 件，每件都标明颜色+材质，含性感/晚装register
- 姿势：新增坐/蹲/躺姿势、半身侧脸/背对镜头/俯仰角度、8个新半身角度光线卡
- **本次最新一轮：再 +90 张卡**（30服装 + 30姿势 + 30 NSFW 状态/场景，仅本地 Klein）
- NSFW 目录加了脸部特写、经典裸体姿势，均保证脸部可见
- 多处"抄参考图背景/服装/裤子鞋子"的提示词漏洞修复

## 生成体验细节

- 队伍里正在真正渲染的那一张卡会高亮（脉冲绿边），跟"排队中"区分开
- 新生成/重新生成的卡片会有红点标记，直到你点开看过
- Regenerate 不再意外跳引擎（原来切了一下引擎复选框会让"重新生成"按钮跟着变）
- 脸部相似度评分现在对比**每一张**参考照片，不只是第一张，取最佳匹配

## 杂项修复

- Krea 配置从 v3 迁移到 v4 时，不再覆盖用户已经自定义过的 `grounding_px`
- 本地 ai-toolkit 训练子进程也带上 `PYTHONUTF8`，避免非 UTF-8 Windows locale 下的编码问题
- ComfyUI 卡死的 barrier 原来会永久阻塞，现在能自动核实并解除（`fix(job-queue)`，这是本次 `fix(queue)` 孤儿行修复之前的第一版）

## 本次会话额外修复（合并/测试基础设施）

跟上游功能无关，是这次把 139 个本地 commit 跟上游 55+ 个 commit 合并时顺手挖出/修的：

- **`discard_orphan_comfyui_barrier` 孤儿队列行 bug**：卡片被删但底层生成任务还卡在 `stalled`，"确认重启"按钮永远清不掉，现在能正确识别并终结这类孤儿行
- **测试套件配置隔离漏洞**：`test_comfyui_control.py` 里一个测试没走隔离夹具，直接写真实 `config.json`，每次跑测试套件都会把用户的 ComfyUI 安装路径覆盖成 pytest 临时路径——已加全局 autouse 夹具堵死这类问题
- **Ollama GPU fence 测试未隔离**：上游新加的本地训练前置检查会真的打一次 Ollama 的 `/api/ps`，开发机上如果真有 Ollama/LDS 在跑，测试会假阳性拒绝——已加全局默认 mock

## 待实现（已有设计，未写代码）

- **图片锁定**：卡片加锁定开关，锁定后单张删除/批量删除/purge 自动跳过，整集删除若含锁定图则整体拒绝
- **数据集设置入口迁移**：数据集列表页每张卡片直接加 ⚙️ 按钮开设置弹窗（改名字/trigger/prompt suffix），不用先进工作区点"⋯ More"

设计文档在 `docs/superpowers/specs/2026-08-02-image-lock-and-settings-entry-design.md`（该目录被 gitignore，仅本地）。
