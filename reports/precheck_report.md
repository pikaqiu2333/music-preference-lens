# Faithfulness Probe Pre-Check

Cases: 3

## Counterfactual Coverage

- constraint_flip: 3
- context_flip: 3
- preference_ablation: 3

## Case Factor Support

### night_drive_rain_electronic_001

- ok: 雨夜开车 (overlap: 夜, 开, 车, 雨)
- ok: 孤独但不压抑 (overlap: 不, 但, 压, 孤, 抑, 独)
- ok: 不要太吵 (overlap: 不, 吵, 太, 要)
- ok: 电子/氛围偏好 (overlap: 围, 子, 氛, 电)

### study_focus_chinese_vocal_001

- ok: 深夜学习 (overlap: 习, 夜, 深)
- check: 低打扰 (overlap: none)
- ok: 中文女声偏好 (overlap: 中, 声, 女, 好, 文)
- ok: 歌词不要抢注意力 (overlap: 不, 力, 意, 抢, 歌, 注, 要, 词)

### party_warmup_discovery_001

- ok: 聚会预热 (overlap: 会, 聚)
- ok: groove (overlap: groove)
- ok: 不要过硬 (overlap: 不, 硬, 要, 过)
- ok: 愿意探索小众 (overlap: 众, 小, 意, 愿, 探, 索)

## Notes

This is only a lexical pre-check. A factor with weak lexical overlap can still be valid, and a factor with strong overlap can still be unfaithful in model output. The next step is to call an LLM and compare original versus counterfactual rankings.
