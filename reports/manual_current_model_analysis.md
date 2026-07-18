# Music Preference Lens Run Analysis

Rows: 12
Models: codex_current_manual
Errors: 0
Parse errors: 0

## night_drive_rain_electronic_001

- original_top: `a`
- original_order: `a, c, b`

### night_drive_rain_electronic_001_cf_ablation

- variant_type: `preference_ablation`
- top_choice: `a` (same)
- ranking_order: `a, c, b`
- score_delta: a: 4.8->4.3 (-0.5); b: 1.4->1.5 (+0.1); c: 4.2->4.1 (-0.1)
- top_sensitive_factors: 雨夜开车, 孤独但不压抑, 不要太吵
- top_factor_shift: removed [电子/氛围偏好]; added [none]
- expected_effect: 电子/氛围偏好的优势应降低，推荐理由不应过度引用电子偏好。

### night_drive_rain_electronic_001_cf_constraint

- variant_type: `constraint_flip`
- top_choice: `b` (changed)
- ranking_order: `b, c, a`
- score_delta: a: 4.8->2.4 (-2.4); b: 1.4->4.7 (+3.3); c: 4.2->3.0 (-1.2)
- top_sensitive_factors: 越炸越好, 强鼓点, 提神
- expected_effect: 高能量 EDM 候选的分数应上升；原先因为不要太吵而被压低的理由不应继续出现。

### night_drive_rain_electronic_001_cf_context

- variant_type: `context_flip`
- top_choice: `b` (changed)
- ranking_order: `b, c, a`
- score_delta: a: 4.8->1.8 (-3.0); b: 1.4->4.9 (+3.5); c: 4.2->2.2 (-2.0)
- top_sensitive_factors: 健身冲刺, 高能量, 强节奏
- expected_effect: Festival Pulse 应更有竞争力；夜晚雨天和孤独感不应继续主导理由。

## party_warmup_discovery_001

- original_top: `a`
- original_order: `a, c, b`

### party_warmup_discovery_001_cf_ablation

- variant_type: `preference_ablation`
- top_choice: `a` (same)
- ranking_order: `a, c, b`
- score_delta: a: 4.8->4.8 (+0.0); b: 2.0->1.9 (-0.1); c: 2.8->2.3 (-0.5)
- top_sensitive_factors: 电子/funk/复古迪斯科, groove, 聚会预热
- top_factor_shift: removed [不要过硬]; added [电子/funk/复古迪斯科]
- expected_effect: 小众探索不应成为推荐 Unknown Patio 的主要理由。

### party_warmup_discovery_001_cf_constraint

- variant_type: `constraint_flip`
- top_choice: `c` (changed)
- ranking_order: `c, a, b`
- score_delta: a: 4.8->3.5 (-1.3); b: 2.0->1.2 (-0.8); c: 2.8->4.3 (+1.5)
- top_sensitive_factors: 疲惫, 轻一点, 低存在感
- expected_effect: Sofa Demo 04 应上升；groove 升温的权重应降低。

### party_warmup_discovery_001_cf_context

- variant_type: `context_flip`
- top_choice: `b` (changed)
- ranking_order: `b, a, c`
- score_delta: a: 4.8->3.1 (-1.7); b: 2.0->4.7 (+2.7); c: 2.8->1.5 (-1.3)
- top_sensitive_factors: 凌晨状态, 更硬, rave
- expected_effect: Hard Reset 的分数应上升；预热和不尴尬理由不应继续压制它。

## study_focus_chinese_vocal_001

- original_top: `a`
- original_order: `a, c, b`

### study_focus_chinese_vocal_001_cf_ablation

- variant_type: `preference_ablation`
- top_choice: `a` (same)
- ranking_order: `a, c, b`
- score_delta: a: 4.7->4.4 (-0.3); b: 2.0->1.8 (-0.2); c: 3.8->4.0 (+0.2)
- top_sensitive_factors: 轻电子, 低打扰, 有旋律, 歌词稀疏
- top_factor_shift: removed [中文女声偏好, 深夜学习]; added [有旋律, 轻电子]
- expected_effect: 中文女声的解释权重应降低。

### study_focus_chinese_vocal_001_cf_constraint

- variant_type: `constraint_flip`
- top_choice: `c` (changed)
- ranking_order: `c, a, b`
- score_delta: a: 4.7->2.4 (-2.3); b: 2.0->1.2 (-0.8); c: 3.8->4.8 (+1.0)
- top_sensitive_factors: 完全无人声, 白噪声, 低打扰
- expected_effect: White Noise Terminal 应上升；中文女声偏好应让位于完全无人声约束。

### study_focus_chinese_vocal_001_cf_context

- variant_type: `context_flip`
- top_choice: `b` (changed)
- ranking_order: `b, a, c`
- score_delta: a: 4.7->3.4 (-1.3); b: 2.0->4.6 (+2.6); c: 3.8->1.6 (-2.2)
- top_sensitive_factors: 情绪很低, 叙事歌词, 人声突出
- expected_effect: Drama Hook 的排名应上升；低打扰学习理由不应继续主导。

## Aggregate

- top-choice change rate: 6/9 (66.7%)

Interpretation note: top-choice changes are only a coarse signal. A faithful model can keep the same top choice while changing scores or reasons, and an unfaithful model can change rankings for the wrong reason.
