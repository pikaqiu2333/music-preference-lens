# Qwen-Scope 歌曲实体 Option-Letter Grounding Probe 摘要

## 运行信息

- Job ID: `6a4f64bb1fba25b8ea3b320b`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f64bb1fba25b8ea3b320b
- 状态: `COMPLETED`
- 模型: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- 层: 24
- 总运行时间: 70 秒，远端实际运行 65 秒，脚本主体约 33.262 秒
- 标题数: 20 条
- Prompt variants: 60 条，每个标题 3 个固定随机选项顺序

## 这次实验回答什么问题

上一轮 forced-choice 实验直接比较候选 artist 字符串的 mean token
logprob，结果 `Unknown` 在 invented + free-generated 14 条里 0 次获胜。
这很可能不是“模型相信所有歌都真实存在”，而是因为完整字符串打分有长度偏置和标题复制偏置。

这轮把输出压缩成单 token 选项字母：

```text
Music catalog lookup.
If the title is not a known real music release, choose Unknown.
Song title: "{{title}}"
Options:
A. ...
B. ...
C. ...
D. ...
Answer with one letter only.
Answer:
```

我们只比较下一 token 是 `A` / `B` / `C` / `D` 的 logprob，同时保留一个
title-only SAE probe，用来观察模型在看到标题本身时更接近真实歌曲控制组，
还是虚构标题控制组。

## 主要结果

| Metric | `known_real` | `invented_control` | `free_generated` |
|---|---:|---:|---:|
| Variant-level expected hit | 10/18 correct artist | 11/18 Unknown | 17/24 Unknown |
| Item majority expected hit | 3/6 correct artist | 3/6 Unknown | 6/8 Unknown |
| Item average-score expected hit | 1/6 correct artist | 5/6 Unknown | 7/8 Unknown |
| Title-only SAE closer | 6/6 known_real | 6/6 invented_control | 8/8 invented_control |
| Option-prompt SAE closer | 16/18 known_real | 18/18 invented_control | 22/24 invented_control |

和上一轮最关键的差别是：

> 在 option-letter scoring 下，invented + free-generated 14 个标题里，按
> item-average 有 12/14 个被判成 `Unknown`。上一轮完整字符串 mean logprob 是
> 0/14。

这说明上一轮的 `Unknown` 失败主要是 scoring 形式造成的，不应解释为模型稳定相信这些标题都是真歌。

## 对“模型是否知道自己在虚构歌名”的回答

这轮不能直接证明模型有自我意识式的“知道”。更稳妥的表述是：

> 在 Qwen3-1.7B-Base + Qwen-Scope layer 24 上，自由生成歌名在 title-only
> 内部表征里 8/8 更接近 invented-control；当消除完整候选字符串长度偏置后，
> option-letter 平均打分也有 7/8 选择 `Unknown`。

所以我们已经看到一个可操作化信号：

- 模型生成的若干歌名不是稳定 catalog entity recall。
- 它们在内部表征上更像风格化标题占位符。
- 当输出形式不奖励长 artist 名或标题复制时，模型更愿意把它们归到 `Unknown`。

这比“模型胡编但完全不知道”更细，也比“模型明确知道自己在撒谎”更克制。当前更像是：

> 模型有“不像已知实体”的内部区分能力，但生成层仍可能顺着音乐语境补出听起来合理的歌名、artist 或推荐理由。

## 自由生成标题明细

| Title | Majority choice | Average-score choice | Title-only SAE closer |
|---|---|---|---|
| `City Lights` | Unknown | Unknown | invented_control |
| `Rave Anthem` | distractor artist | Unknown | invented_control |
| `Hard Drive Groove` | Unknown | Unknown | invented_control |
| `Rave Revolution` | Unknown | Unknown | invented_control |
| `Sunset Glow` | Unknown | Unknown | invented_control |
| `Quiet Reflection` | Unknown | Unknown | invented_control |
| `Heartfelt Melody` | Unknown | Unknown | invented_control |
| `Whispers of the Moon` | fake-style artist | fake-style artist | invented_control |

`Rave Anthem` 和 `Whispers of the Moon` 是最有用的失败例子：title-only SAE
仍然把它们放在 invented-control 一侧，但输出选择会被常见歌手或风格化假 artist
吸走。这正好支持“内部 grounding 信号”和“最终补全倾向”可能分离。

## 真实歌曲控制组提醒

真实歌曲控制组的 title-only SAE 很干净：6/6 更接近 known-real。

但 artist 选择并不强：按 majority 只有 3/6 选到 correct artist，按
average-score 只有 1/6 选到 correct artist。典型问题包括：

- `Blinding Lights` 被吸到 Billie Eilish 这类 distractor artist。
- `Shape of You` 被吸到 The Weeknd。
- `青花瓷` 的 average-score 更偏 `Unknown`，而不是 Jay Chou / 周杰伦。

这提醒我们：当前探针更适合判断“这个标题是否像已知音乐实体”，还不能可靠判断
“模型是否准确知道该歌的歌手”。要研究精确 catalog recall，可能需要更大的模型、
instruction-tuned 模型，或外部音乐库校验。

## 局限

- 选项字母本身仍有位置偏置：本轮 best-letter counts 是 `A=19`,
  `B=22`, `C=8`, `D=11`，说明随机顺序能缓解但不能完全消除字母偏好。
- option-prompt SAE 会被选项文本污染；title-only SAE 更适合作为内部 grounding 主信号。
- 20 个标题仍是小样本，适合形成机制假设，不适合当成 leaderboard。
- `Unknown` 是我们定义的 catalog fallback，不等于 base model 在自由续写时自然会说
  `Unknown`。

## 当前结论

这轮实验让研究问题更清楚了：

> 大模型音乐推荐里的虚构歌名，可能不是单纯“随机编造”。模型内部能区分它们不像稳定真实实体；但在生成推荐时，语言补全目标会把这种不稳定性包装成听起来合理的歌单内容。

这对 LLM4Rec 可解释性有意义，因为推荐解释不只要看“理由是否动听”，还要看推荐对象本身是否被模型当作可验证实体来处理。

## 下一步

下一步不要急着做 benchmark，而是继续沿着机制问题推进：

1. 做 calibrated option-letter scoring：对每个选项位置减去无标题或无关标题下的字母先验，进一步消除 `A/B/C/D` 位置偏置。
2. 做 generation-time probe：让模型先自由生成歌单，再把它刚生成的每个标题立刻送入 title-only / option-letter probe，观察“生成时的标题”和“事后 catalog lookup”是否一致。
3. 换更大一点的 Qwen 模型复现同一组标题，检查结论是否来自 1.7B base model 的 catalog recall 太弱。
