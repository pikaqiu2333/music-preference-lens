# Qwen-Scope 自由歌单生成 Probe 摘要

## 运行信息

- Job ID: `6a4f4e881fba25b8ea3b308f`
- Job URL: https://huggingface.co/jobs/REDACTED/6a4f4e881fba25b8ea3b308f
- 状态: `COMPLETED`
- 模型: `Qwen/Qwen3-1.7B-Base`
- SAE: `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50`
- 层: 24
- 总运行时间: 120 秒，远端脚本主体运行约 84 秒
- Prompt rows: 8 条，4 组 original / counterfactual

## 这次实验回答什么问题

这次不再给候选歌曲，也不再指定 `Best track`。Prompt 只给：

```text
User profile: ...
Current need: ...
Generate a 5-track playlist...
Playlist:
1. Title:
```

所以它更接近真实的生成式推荐产品问题：

> 当用户当前需求改变时，模型自由生成的歌单内容和推荐理由是否跟着改变？对应的 SAE 特征是否也同步变化？

## 主要结果

这次自由生成成功。模型在 4 组反事实里都生成了有意义的歌单风格变化。

答案 token 上，12 个预设 feature 假设中有 11 个方向对齐。

| Probe | 内容变化 | Feature | 预期 | Answer Delta | 对齐 |
|---|---|---|---|---:|---|
| `free_night_drive_to_rave` | 雨夜安静驾驶 -> 高能 rave | `feature32078` energy/activation | 上升 | `+61.648` | Yes |
| `free_night_drive_to_rave` | 雨夜安静驾驶 -> 高能 rave | `feature7884` rave/intensity | 上升 | `+74.273` | Yes |
| `free_night_drive_to_rave` | control | `feature30224` identity control | 近零 | `-0.040` | Yes |
| `free_peak_party_to_soft_afterparty` | peak dance floor -> soft after-party | `feature32078` energy/activation | 下降 | `-40.412` | Yes |
| `free_peak_party_to_soft_afterparty` | peak dance floor -> soft after-party | `feature30584` quiet/low presence | 上升 | `+32.918` | Yes |
| `free_peak_party_to_soft_afterparty` | control | `feature19603` identity control | 近零 | `+0.066` | Yes |
| `free_study_to_emotional_vocal` | 写作背景 -> 情绪人声叙事 | `feature8706` emotional vocal/lyrics | 上升 | `+24.312` | Yes |
| `free_study_to_emotional_vocal` | 写作背景 -> 情绪人声叙事 | `feature24979` focus/low distraction | 下降 | `-5.946` | Yes |
| `free_study_to_emotional_vocal` | control | `feature22232` identity control | 近零 | `0.000` | Yes |
| `free_emotional_vocal_to_no_vocals` | 情绪人声 -> 无人声写作 | `feature8706` emotional vocal/lyrics | 下降 | `-29.898` | Yes |
| `free_emotional_vocal_to_no_vocals` | 情绪人声 -> 无人声写作 | `feature24979` no vocals/focus | 上升 | `+5.964` | Yes |
| `free_emotional_vocal_to_no_vocals` | 情绪人声 -> 无人声写作 | `feature24916` instrumental/no vocals | 上升 | `-0.288` | No |

## 生成例子

### 雨夜驾驶 -> 高能 Rave

原始需求是安静雨夜驾驶。模型生成了类似：

```text
Title: "City Lights"
Sound: Ambient, electronic, and rhythmic indie pop.
Reason: ... calming urban mood ... restrained beat ... not too loud ...
```

反事实需求改成 peak-time rave session 后，模型生成了类似：

```text
Title: "Rave Anthem"
Sound: High-energy techno with a driving beat and a pulsating bassline.
Reason: ... peak-time rave session ... energy and pressure ...
```

对应 answer-token 特征变化：

- `feature32078` energy/activation: `+61.648`
- `feature7884` rave/intensity: `+74.273`
- `feature30224` identity control: `-0.040`

### 情绪人声 -> 无人声写作

原始需求是 late-night emotional listening，模型生成了带人声和情绪表达的内容：

```text
Title: "Heartfelt Melody"
Sound: Chinese female vocals with a heartfelt and emotive delivery.
Reason: ... heartfelt vocal and lyrical story ...
```

反事实需求改成 strict writing mode 后，模型生成了：

```text
Title: "Whispers of the Moon"
Sound: Soft electronic music with a gentle, ambient texture.
Reason: ... absence of vocals and lyrics ... focus on the instrumental elements ...
```

对应 answer-token 特征变化：

- `feature8706` emotional vocal/lyrics: `-29.898`
- `feature24979` no vocals/focus: `+5.964`
- `feature24916` instrumental/no vocals: `-0.288`

这里最后一个更细的 instrumental/no-vocals feature 没有对齐，但主 no-vocals/focus feature 对齐了。

## 解释

这次实验比之前的 `Best track` 控制实验更接近我们真正想研究的场景：

> 模型不是在解释我们指定的选择，而是在自己生成一个新歌单。

目前可以谨慎地说：

> 在 Qwen3-1.7B-Base + Qwen-Scope layer 24 上，用户需求的反事实变化会同时反映在自由生成歌单文本和相关音乐偏好 SAE feature 的 answer-token 激活变化中。

但还不能说：

> 这些 feature 因果决定了模型推荐什么歌。

因为我们还没有做 feature intervention，也没有把生成内容拆成更细的 span 来定位每个短语的激活。

## 局限

- 模型可能生成虚构歌名；本实验不评价歌名事实性。
- Prompt 结尾是 `1. Title:`，模型开头先补了一段空的 `Sound/Reason` 模板，然后才生成 `## Playlist`。这说明 prompt 还可以清理。
- 本次 answer feature 是对整段生成文本求均值，包含标题、sound、reason 和格式 token；下一步应该只测推荐理由 span 或音乐描述 span。
- 结果仍然是相关性，不是因果性。

## 下一步

1. 改 prompt，减少开头模板重复，例如改成 `Playlist:\n1.` 或更短的自然续写格式。
2. 做 generated-span probe：把生成结果切成 `Title` / `Sound` / `Reason` span，只在相关 span 上测 SAE feature。
3. 加中文版本 prompt，测试中文音乐推荐语里这些 feature 是否仍然可观察。
4. 如果 span 结果稳定，再考虑 feature intervention，而不是现在就做大 benchmark。
