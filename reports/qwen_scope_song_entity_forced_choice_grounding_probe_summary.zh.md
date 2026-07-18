# Qwen-Scope 歌曲实体 Forced-Choice Grounding Probe 摘要

## 运行信息

- Job ID: `6a4f60381499512f2377a20e`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f60381499512f2377a20e
- 状态: `COMPLETED`
- 模型: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- 层: 24
- 总运行时间: 69 秒，远端实际运行 64 秒，脚本主体约 32 秒
- Prompt rows: 20 条

## 这次实验回答什么问题

上一轮 open generation probe 会让模型从 `Artist:` 漂到 `Album:`。这轮把问题改成 forced-choice / logprob：

```text
Music catalog lookup.
If the title is not a known real music release, answer Unknown.
Song title: "{{title}}"
Artist:
```

每个标题都有 3-4 个候选 artist / `Unknown`。我们测两件事：

- 候选 continuation 的 mean token logprob，看看模型更愿意接哪个答案。
- `Artist:` 后的 prompt-final SAE 激活更接近真实歌曲控制组，还是虚构标题控制组。

## 主要结果

| Group | Expected hit | Best type distribution | Prefix SAE closer | Best-answer SAE closer |
|---|---:|---|---|---|
| `known_real` | 4/6 | 4 correct, 2 distractor | 6/6 known_real | 6/6 known_artist |
| `invented_control` | 0/6 | 5 distractor, 1 fake_style | 6/6 invented_control | 5/6 known_artist |
| `free_generated` | 0/8 | 4 distractor, 4 fake_style | 8/8 invented_control | 5/8 known_artist |

最值得信的是 prompt-final SAE：

> free-generated 标题 8/8 都更接近 invented_control，而不是 known_real。

这比上一轮更强地支持一个判断：

> 模型在准备回答之前，内部状态把这些自由生成标题更像“未验证 / 虚构标题”，而不是稳定真实歌曲实体。

但 mean logprob 的候选选择不可靠：

> invented + free-generated 一共 14 条，`Unknown` 0 次成为最高 mean logprob。

这不能直接解释成“模型相信它们都是真歌”。更可能的原因是 scoring 方法有偏差：mean token logprob 会奖励多 token artist 名、标题复制和风格化假 artist，而 `Unknown` 只有一个 token。

## 真实歌曲控制组

| Title | Best candidate | Hit | Margin vs best other | Prefix closer |
|---|---|---|---:|---|
| `Blinding Lights` | distractor artist, detailed row truncated by HF log tail | No | - | known_real |
| `bad guy` | Billie Eilish | Yes | +0.805 | known_real |
| `Shape of You` | Ed Sheeran | Yes | +1.403 | known_real |
| `Bohemian Rhapsody` | Ed Sheeran | No | -3.467 | known_real |
| `稻香` | 周杰伦 | Yes | +1.660 | known_real |
| `青花瓷` | 周杰伦 | Yes | +0.751 | known_real |

`Bohemian Rhapsody` 是一个很好的方法提醒：按 mean logprob，`Ed Sheeran` 赢了；但按 total logprob，`Queen` 其实比 `Ed Sheeran` 更高。这说明当前 forced-choice 的归一化方式会改变结论。

## 自由生成标题

| Title | Best candidate type | Best candidate | Prefix closer | Best-answer closer |
|---|---|---|---|---|
| `City Lights` | distractor_artist | The Weeknd | invented_control | known_artist |
| `Rave Anthem` | distractor_artist | Billie Eilish | invented_control | known_artist |
| `Hard Drive Groove` | fake_style_artist | Hard Drive Groove | invented_control | unknown_answer |
| `Rave Revolution` | fake_style_artist | Rave Revolution | invented_control | known_artist |
| `Sunset Glow` | distractor_artist | Billie Eilish | invented_control | known_artist |
| `Quiet Reflection` | fake_style_artist | Quiet Reflection | invented_control | unknown_answer |
| `Heartfelt Melody` | fake_style_artist | Heartfelt Melody | invented_control | unknown_answer |
| `Whispers of the Moon` | distractor_artist | Billie Eilish | invented_control | known_artist |

这个表要分开读：

- Prefix closer 是在模型还没输出候选答案前测的，更接近“它如何表征这个标题”。
- Best candidate 是按 mean logprob 选出来的，容易被候选文本长度和复制标题影响。
- Best-answer closer 主要受最终候选文本本身影响，所以不如 prefix signal 干净。

## 解释

这轮结果不是简单地证明“模型知道自己在虚构歌名”。更准确的说法是：

> 在 Qwen3-1.7B-Base + Qwen-Scope layer 24 上，模型的 prompt-final 内部状态能把 known-real、invented-control 和 free-generated 标题分开；其中 free-generated 标题 8/8 更接近 invented-control。  
> 但当前候选 logprob scoring 还不能可靠表达“模型是否会主动回答 Unknown”，因为 mean token logprob 有长度和复制偏差。

这其实是一个很有价值的研究发现：内部表征和输出选择会分离。

在产品语言里可以理解成：

- 模型内部可能知道这个标题“不像稳定 catalog entity”。
- 但生成/打分层仍然会顺着音乐语境补一个听起来合理的 artist 或项目名。
- 所以推荐解释里的“歌名 + 推荐语”即使流畅，也可能不是可验证的实体推荐。

## 局限

- HF Jobs MCP 日志只返回最后 20 行，`Blinding Lights` 的详细 row 被截断；但汇总里包含它，并显示 known-real 共有 2 个 miss。
- 当前主排序用 mean token logprob，会偏向长候选和标题复制。
- `Unknown` 是人为引入的 catalog fallback，不一定是 base model 自然会续写的答案。
- Best-answer SAE 被候选答案文本强烈污染，不适合作为主要 grounding 判断。

## 下一步

下一轮应该改成 option-letter scoring：

```text
Song title: "Hard Drive Groove"
Options:
A. Unknown
B. Ed Sheeran
C. Queen
D. Hard Drive Groove
Answer:
```

只比较 `A` / `B` / `C` / `D` 单 token logprob，并随机打乱选项顺序跑 3 个 seed。这样可以消掉 artist 名长度、标题复制和 `Unknown` 单 token 的归一化问题。

后续主要看：

- free-generated 是否仍然 8/8 prefix closer to invented_control。
- option-letter 是否把 invented/free 更多判成 `Unknown`。
- 如果 prefix 表征和 option-letter 输出仍分离，这就是一个很清楚的“内部不稳定实体感知 vs 输出补全倾向”的研究点。
