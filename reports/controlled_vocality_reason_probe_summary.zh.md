# 受控有人声 / 纯器乐理由实验（大白话版）

## 一句话结论

不使用 Song ID、只使用真实歌名和歌手名，也能得到一个干净的因果结果：

- 在固定的 8 个真实候选中，模型能稳定根据“有人声”或“纯器乐”理由把概率移向正确类别。
- 在让模型自由续写固定 title-artist 的 teacher-forced 分数中，8 首有 6 首方向正确。
- Space Oddity 和 Awake 是两个方向错误的关键失败案例。

因此，模型具备使用 vocality 属性做比较的能力，但这种能力没有在所有生成式 title-artist 续写中稳定生效。

## 数据为什么比上一轮干净

本轮只有 8 首歌：

### 纯器乐

- An Ending (Ascent) - Brian Eno
- River Flows in You - Yiruma
- Awake - Tycho
- Merry Christmas, Mr. Lawrence - Ryuichi Sakamoto

### 有人声

- Space Oddity - David Bowie
- Blinding Lights - The Weeknd
- All of Me - John Legend
- Don't Stop Me Now - Queen

所有 title-artist 都同时通过 MusicBrainz 和 Apple 精确核验，每位艺术家只出现一次。是否有人声来自独立页面、官方乐谱或明确演奏/主唱 credits，而不是模型自己判断。

三条理由都正好 11 个英文词，不出现任何歌名或歌手：

- vocal：The track features prominent vocals and clearly audible sung lyrics throughout.
- instrumental：The track features no vocals and only instrumental musical passages throughout.
- neutral：The track has musical qualities that may suit this listening request.

## 实验一：固定实体分数

每首歌分别放在 vocal、instrumental 和 neutral 理由之后，计算完整歌名加歌手的平均对数概率。

匹配理由减去反转理由的结果：

| Track | 真实属性 | Matched margin | 方向 |
|---|---|---:|---|
| An Ending (Ascent) - Brian Eno | instrumental | 0.270 | 正确 |
| River Flows in You - Yiruma | instrumental | 0.251 | 正确 |
| Merry Christmas, Mr. Lawrence - Ryuichi Sakamoto | instrumental | 0.124 | 正确 |
| Awake - Tycho | instrumental | -0.153 | 错误 |
| All of Me - John Legend | vocal | 0.297 | 正确 |
| Blinding Lights - The Weeknd | vocal | 0.204 | 正确 |
| Don't Stop Me Now - Queen | vocal | 0.211 | 正确 |
| Space Oddity - David Bowie | vocal | -0.147 | 错误 |

总体 6/8 正确，刚好达到预注册的 75% 门槛。两组中位 margin 都为正。

注意：多数 matched reason 并没有胜过 neutral reason。这里能证明的是 vocality 词语翻转会按正确方向改变 6/8 个 pair 的相对支持，不能说写属性理由总会提高歌曲绝对概率。

## 实验二：固定候选选择

把 8 首真实歌曲一起列出，让模型在理由之后输出 A-H。候选顺序运行两次并完全反转，以减少字母和位置偏差。

平均“vocal 候选总 logits - instrumental 候选总 logits”：

- vocal reason：1.330
- neutral reason：-0.651
- instrumental reason：-1.233

相对 neutral：

- vocal reason 向 vocal 候选移动 1.982。
- instrumental reason 向 instrumental 候选移动 0.582。

两个候选顺序中方向都正确：

- vocal reason 的最高项分别是 All of Me 和 Space Oddity。
- instrumental reason 的最高项分别是 River Flows in You 和 An Ending (Ascent)。

## Space Oddity 为什么重要

孤立 pair 续写时：

- vocal reason 下：-2.519
- instrumental reason 下：-2.372

模型反而在 instrumental reason 下更支持 Space Oddity，方向错误。

但在 8 首真实候选中比较时，vocal reason 又能把 Space Oddity 选成最高项之一。

这说明问题不是简单的“模型完全不知道 Space Oddity 有人声”。更可能是：

1. 比较真实候选时，模型能够调用 vocality 知识。
2. 自由续写 title-artist 时，歌曲名称先验、文字关联或局部生成路径可能盖过这个属性。
3. 同一个模型知识在“比较”与“生成”两种 harness 中没有稳定转化为一致行为。

这正是 Agent Harness / 推荐架构有意义的地方：给模型合法候选进行比较，可能比让它直接生成实体更可靠。

## Awake 的含义

Awake - Tycho 的 matched margin 为 -0.153，也错误，但固定候选下 instrumental reason 的最高项不是 Awake，而是 River Flows in You 或 An Ending (Ascent)。

这可能表示模型对 Tycho - Awake 的具体 vocality 记忆较弱，或者 Awake 这个普通词带来了较强名称先验。它与 Space Oddity 应作为两种不同失败：

- Space Oddity：候选比较正确、生成式 pair 分数错误，像控制或 harness 问题。
- Awake：两种任务都没有显示该实体是最强 instrumental 代表，像具体实体知识不足。

## 研究决策

预注册行为门槛通过，可以进入机制实验，但不能只看成功平均值。

下一步使用四类对照：

- vocal success：Blinding Lights
- vocal generation failure：Space Oddity
- instrumental success：River Flows in You
- instrumental weak/failure：Awake

在第 14-28 层移植 vocal 与 instrumental 理由状态，分别观察：

- 固定 pair 分数在哪里开始翻转；
- 固定候选类别 logits 在哪里开始翻转；
- attention 与 MLP 对成功和失败案例是否不同。

如果 Space Oddity 的正确 vocal 信号在候选比较中存在、在生成路径中被覆盖，就能更直接支持“模型知道部分属性，但 harness 没有稳定使用”的控制缺口假设。

