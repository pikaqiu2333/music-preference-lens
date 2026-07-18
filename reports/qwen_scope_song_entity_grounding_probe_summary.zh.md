# Qwen-Scope 歌曲实体 Grounding Probe 摘要

## 运行信息

- Job ID: `6a4f59d71fba25b8ea3b3156`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f59d71fba25b8ea3b3156
- 状态: `COMPLETED`
- 模型: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- 层: 24
- 总运行时间: 268 秒，远端实际运行 131 秒，脚本主体约 96 秒
- Prompt rows: 20 条

## 这次实验回答什么问题

上一轮自由歌单生成里，模型没有候选歌曲列表，也没有我们指定的
`Best track`。它自己生成了 `City Lights`、`Rave Anthem`、`Heartfelt
Melody` 这类标题。

这带来一个更细的问题：

> 模型生成这些歌名时，是在召回预训练里学到的真实歌曲实体，还是在按音乐风格编一个“像歌名的标题”？

这次 probe 用一个很短的 catalog-style prompt 来测：

```text
Music catalog lookup.
Song title: "{{title}}"
Artist:
```

我们把标题分成三组：

- `known_real`: 真实歌曲控制组，例如 `Shape of You`、`Bohemian Rhapsody`、`稻香`。
- `invented_control`: 人工编的控制标题，例如 `Neon Rain Avenue`、`Soft Circuit Dream`。
- `free_generated`: 上一轮自由歌单生成里的标题，例如 `City Lights`、`Rave Anthem`。

然后观察两件事：

- 模型补出来的 artist/catalog 信息是否稳定、是否命中真实 artist。
- 这些 prompt span 和 answer span 的 SAE 表征更接近真实歌曲控制组，还是更接近虚构标题控制组。

## 主要结果

第一，Qwen3-1.7B-Base 的 catalog lookup 能力本身不强。6 个真实歌曲控制里只命中 3 个，准确率是 `3/6 = 50%`。

第二，虚构标题控制组很干净。6 个 invented controls 在 prompt span 上全部更接近 `invented_control` prototype。

第三，自由生成标题最有意思：

- 只看标题 prompt span，8 个自由生成标题里有 4 个更接近真实歌曲控制组，4 个更接近虚构标题控制组。
- 但看模型补 artist/catalog 的 answer span，8 个里有 7 个更接近虚构标题控制组。

也就是说：

> 自由生成出的歌名，单看标题有时会像真实歌曲实体；但当模型真的要补 artist/catalog 信息时，它大多表现得更像在处理一个虚构标题。

## 真实歌曲控制组

| Title | Expected artist | Greedy completion | Hit | Consistency |
|---|---|---|---|---:|
| `Blinding Lights` | The Weeknd | `The Beatles...` | No | 0.25 |
| `bad guy` | Billie Eilish | album-style wrong continuation | No | 0.50 |
| `Shape of You` | Ed Sheeran | `Ed Sheeran...` | Yes | 0.75 |
| `Bohemian Rhapsody` | Queen | `Queen...` | Yes | 0.75 |
| `稻香` | 周杰伦 / Jay Chou | `周杰伦...` | Yes | 0.25 |
| `青花瓷` | 周杰伦 / Jay Chou | `洛天依...` | No | 0.25 |

这个结果提醒我们：不能把这次 probe 解释成“模型的世界知识很可靠”。它只能作为一个弱模型上的初步内部信号实验。

## 自由生成标题

| Title | Greedy completion pattern | Prompt closer | Answer closer | Consistency |
|---|---|---|---|---:|
| `City Lights` | `Album: "The Sound of Music"...` | known_real | known_real | 0.25 |
| `Rave Anthem` | `Album: "Rave Anthem"...` | invented_control | invented_control | 0.25 |
| `Hard Drive Groove` | `Album: "Hard Drive Groove"...` | invented_control | invented_control | 0.25 |
| `Rave Revolution` | `Album: "Rave Revolution"...` | invented_control | invented_control | 0.25 |
| `Sunset Glow` | `Album: "Sunset Glow"...` | known_real | invented_control | 0.25 |
| `Quiet Reflection` | `Album: "The Quiet Reflection"...` | known_real | invented_control | 0.25 |
| `Heartfelt Melody` | `Album: "Eternal Harmony"...` | known_real | invented_control | 0.25 |
| `Whispers of the Moon` | `Album: "Whispers of the Moon"...` | invented_control | invented_control | 0.25 |

这些 completion 有一个共同点：模型经常没有老老实实补 `Artist:`，而是滑到 `Album:` 或泛化的 catalog metadata。这说明它不是稳定地把标题绑定到某个真实歌曲实体上。

## 解释

目前可以谨慎地说：

> 在 Qwen3-1.7B-Base + Qwen-Scope layer 24 上，上一轮自由生成出来的歌名更像“风格化标题占位符”，而不是稳定的真实歌曲实体召回。模型确实知道部分真实歌曲，但当它面对自己刚生成的泛化标题时，大多数 answer-span 表征更接近虚构标题控制组。

这和我们的产品直觉是对上的：生成式音乐推荐里的“歌名”可能有两种来源。

- 一种是实体召回：模型真的知道 `Shape of You -> Ed Sheeran`。
- 另一种是风格补全：模型根据当前需求生成一个像歌名的短语，例如 `Quiet Reflection`、`Hard Drive Groove`。

这两类表面上都像推荐结果，但机制和可信度完全不同。对 LLM4Rec 可解释性来说，这个区别很重要：如果标题只是风格补全，那么推荐理由再合理，也不等于模型真的在推荐一个可验证的音乐实体。

## 局限

- Qwen3-1.7B-Base 的真实歌曲 artist 命中率只有 50%，所以不能把这次结论推广到更强模型。
- 这次没有做外部音乐数据库验证，只是在模型内部和少量人工控制标题之间比较。
- Prompt 太短，模型经常从 `Artist:` 漂移到 `Album:`，说明 open generation 的格式约束不够强。
- 这仍然是相关性 probe，不是因果干预。

## 下一步

下一步不需要转向大 benchmark。更合适的是做一个更小、更干净的 forced-choice / logprob grounding probe：

```text
Music catalog lookup.
Song title: "Shape of You"
Artist: <candidate>
```

对真实歌曲，比较正确 artist、干扰 artist、`Unknown` 的 logprob 和 SAE 激活。

对自由生成标题和 invented controls，比较 `Unknown`、随机真实 artist、模型自造 artist 的 logprob 和 SAE 激活。

这样可以减少 open generation 的格式漂移，把问题更明确地压缩成：

> 模型是否把这个标题当成一个已知音乐实体来打分？
