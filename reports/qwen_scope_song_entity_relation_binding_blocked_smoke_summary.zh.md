# Qwen-Scope 关系绑定 Block-Held-Out Smoke 报告

## 结论摘要

修复后的 matched relation-binding smoke 显示：

> Qwen3-1.7B-Base 的输出概率能够识别多数著名歌曲的正确标题-歌手关系，但当前
> layer-24 SAE 特征方向不能跨歌曲稳定泛化地读取这种关系知识。

因此本轮不运行 full，不进入自由生成 artist-shuffle，也不做 SAE intervention。

## 实验状态

- Hugging Face Job: `6a50af642055b7ba2bc12ceb`
- 模型: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- 层: 24
- 数据: 6 个独立双向交换块，12 条关系，中英文各 6 条
- 交叉验证: 3 折，每折完整留出一个英文块和一个中文块
- GPU 运行时间: 约 73 秒
- 技术门: 通过

## 预注册结果

| 指标 | 结果 | 门槛 | 状态 |
|---|---:|---:|---|
| Direct PMI-style accuracy | 0.833 (10/12) | 0.80 | 通过 |
| Order-flipped choice accuracy | 0.750 (9/12) | 诊断指标 | 未达 0.80 |
| Artist-end SAE paired CV | 0.500 (6/12) | 诊断指标 | 随机水平 |
| Pair-end SAE paired CV | 0.583 (7/12) | 0.80 | 失败 |

Direct PMI-style 的失败项是 `光年之外 - 邓紫棋` 和 `演员 - 薛之谦`。A/B
choice 的失败项是 `Blinding Lights - The Weeknd`、`bad guy - Billie Eilish`
和 `演员 - 薛之谦`。两种行为方法仍共同说明模型具有明显但不完美的关系知识。

## 折分修复是否有效

有效。第一次 relation-level smoke 的 artist-end 和 pair-end SAE CV 分别是 `0.000`
和 `0.133`，几乎所有 margin 都反向。改成独立双向交换块并按整块留出后，两者恢复
到 `0.500` 和 `0.583`，系统性反向消失。

这确认第一次极低结果主要来自 opposite-label artist leakage，而不是可信的机制
结论。修复后的 pair-end 三折准确率分别为 `0.50`、`0.75`、`0.50`，仍未表现出
稳定的跨实体方向。

## 可解释性含义

本轮把两个问题分开了：

1. **模型有没有关系知识？** 有。先验校正后的条件概率在 10/12 个关系上偏向正确
   歌手。
2. **一个共享 SAE 方向能否读出关系知识？** 当前不能。layer-24、pair-end、32 个
   监督筛选稀疏特征的组合只达到 7/12。

训练全集上的 margin 大多为正，而 block-held-out 明显下降，提示当前特征更像
实体或局部模式，而不是可跨歌曲复用的 `title -> artist` 关系变量。也可能存在关系
信息是分布式的、出现在其他层，或只在预测 artist token 时被调用。

## 研究决定

不使用相同方法扩大到 20 条 full。扩大四个块不足以把 `0.583` 合理提升到预注册的
`0.80`，而且 full 后再选择特征或层会产生事后调参风险。

下一步应改成无需监督特征选择的内部归因：

1. 在每一层对正确/错误 artist continuation 计算 layerwise logit-lens margin，定位
   关系知识何时进入可解码状态。
2. 对 title token 或关键残差位置做 activation patching，直接测其对正确 artist
   log-prob 的因果贡献。
3. 只在预先定位出的层重新查看 SAE，并使用无监督或独立数据选择特征。

这条路线仍然专注可解释性，不需要扩建推荐 benchmark，也不依赖人工音乐偏好标注。
