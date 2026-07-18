# Qwen-Scope 音乐 Base-Completion 推荐理由探针总结

## 运行信息

- Job ID: `6a4f273a1fba25b8ea3b2e17`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f273a1fba25b8ea3b2e17
- 状态：`complete`
- 基座模型：`Qwen/Qwen3-1.7B-Base`
- SAE 仓库：`Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- 层数：24
- 运行时间：在 `t4-small` 上总计 93 秒
- Prompt 行数：8 条 base-completion prompt

## 目的

这次实验把之前失败的“指令式 JSON 推荐理由 prompt”替换成了更适合 base model 的续写协议：

```text
Best track: B
Reason:
```

探针会预先给定被选中的最佳曲目。模型只需要在 `Reason:` 后面继续写一段简短推荐理由。这样做的目的不是测试模型的排序能力，而是把“推荐理由 token 的生成过程”单独隔离出来。

## 主要结果

这次实验成功了。

模型在 4 组 matched original/counterfactual 对照里都生成了有意义的推荐理由。更重要的是，所有被测试的假设中，answer-token 上的 SAE feature delta 都按预期方向移动，包括 identity/source 负控制特征也保持接近 0。

## 对照结果

| Probe | 理由变化 | 关键特征 | 预期 | Prompt Delta | Answer Delta | 是否对齐 |
|---|---|---|---|---:|---:|---|
| `base_high_energy_night_to_rave` | 安静夜间驾驶 -> 高能量 / rave pressure | `feature32078` 能量/激活 | 正向 | `+29.978` | `+69.192` | 是 |
| `base_high_energy_night_to_rave` | 安静夜间驾驶 -> 高能量 / rave pressure | `feature7884` rave/强度 | 正向 | `+64.729` | `+12.782` | 是 |
| `base_high_energy_night_to_rave` | 控制项 | `feature30224` identity control | 接近 0 | `-0.035` | `0.000` | 是 |
| `base_low_presence_party_to_quiet` | rave pressure -> soft groove / low presence | `feature32078` 能量/激活 | 负向 | `-13.345` | `-17.912` | 是 |
| `base_low_presence_party_to_quiet` | rave pressure -> soft groove / low presence | `feature30584` 安静/柔和语境 | 正向 | `+2.770` | `+15.387` | 是 |
| `base_low_presence_party_to_quiet` | 控制项 | `feature19603` identity control | 接近 0 | `+0.044` | `0.000` | 是 |
| `base_emotional_vocal_study_to_lyrics` | 专注背景音乐 -> 情绪人声 / 叙事 | `feature8706` 情绪人声/歌词 | 正向 | `+25.827` | `+18.356` | 是 |
| `base_emotional_vocal_study_to_lyrics` | 专注背景音乐 -> 情绪人声 / 叙事 | `feature24979` 专注/无人声 | 负向 | `-12.771` | `-10.482` | 是 |
| `base_emotional_vocal_study_to_lyrics` | 控制项 | `feature22232` identity control | 接近 0 | `-0.011` | `+0.148` | 是 |
| `base_no_vocals_lyrics_to_instrumental` | 情绪人声 -> 无人声 / 白噪音专注 | `feature8706` 情绪人声/歌词 | 负向 | `-18.415` | `-28.091` | 是 |
| `base_no_vocals_lyrics_to_instrumental` | 情绪人声 -> 无人声 / 白噪音专注 | `feature24979` 无人声/专注 | 正向 | `+11.048` | `+13.047` | 是 |
| `base_no_vocals_lyrics_to_instrumental` | 情绪人声 -> 无人声 / 白噪音专注 | `feature24916` 器乐/无人声 | 正向 | `+7.703` | `+5.549` | 是 |

## 生成理由示例

### 高能量 / Rave

原始理由：

> Matches user's preference for quiet night drive and calm urban mood.

反事实理由：

> Matches the user's preference for high energy and rave pressure.

匹配到的预期因素：

- 原始：`quiet night drive`，`restrained beat`，`calm urban mood`
- 反事实：`high energy`，`rave pressure`，`strong drums`，`fast activation`

Answer-token 变化：

- `feature32078` 能量/激活：`+69.192`
- `feature7884` rave/强度：`+12.782`
- `feature30224` identity control：`0.000`

### 无人声 / 器乐

原始理由：

> Best emotional match: Drama Hook's emotional story and dramatic chorus align
> with user's need for heartfelt vocal and lyrical story.

反事实理由：

> Best fit for white noise focus: C has a neutral mood, minimal energy, and
> almost no melody, making it ideal for white noise focus.

匹配到的预期因素：

- 原始：`heartfelt vocal`，`lyrical story`，`dramatic chorus`
- 反事实：`no vocals`，`instrumental texture`，`no lyric attention`，`white noise focus`

Answer-token 变化：

- `feature8706` 情绪人声/歌词：`-28.091`
- `feature24979` 无人声/专注：`+13.047`
- `feature24916` 器乐/无人声：`+5.549`

## 解释

这是项目里第一个干净的“生成理由可解释性”结果：

> 当模型被条件化为生成一段简短理由时，answer-token 上的 SAE features 会随着理由中被命名的音乐因素一起移动。

这个结果仍然不是因果性的。我们还没有对 feature 进行干预，也没有证明“改变某个 feature 会改变推荐理由”。但在 matched prompt pairs 下，这是一个很强的相关性对齐结果。

## 为什么重要

上一次 generated-reason 实验失败，是因为 instruction/JSON prompt 让 base model 反复续写 prompt 规则，而不是完成推荐任务。这次实验给出了一个实际可用的方法修正：

- 不要要求 base model 完成指令跟随式 JSON 任务。
- 用一个短的自然语言续写格式来条件化模型。
- 只测量 `Reason:` 后面的 tokens。

这为“不依赖 benchmark 或大规模人工标注”的 explanation-faithfulness 探针提供了一条可行路径。

## 下一步实验

下一步应该把分析做得更局部：

1. 把生成理由拆成 phrase spans，例如 `high energy`、`rave pressure`、`heartfelt vocal` 和 `no vocals`。
2. 不只测量整个 answer，而是测量这些生成 phrase spans 上的 feature activation。
3. 加入同样 4 组 probe 的中文版本。
4. 继续使用 identity/source features 作为负控制。

当前结论：

> Base-completion 协议让 Music Preference Lens 真正变成了一个 generated-reason interpretability experiment。现在我们在同一组 matched probes 里同时拥有 prompt-side feature movement、生成理由文本，以及 answer-token feature movement 的对齐证据。
