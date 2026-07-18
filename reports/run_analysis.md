# Music Preference Lens Run Analysis

Rows: 12
Models: gpt-5.5
Errors: 12
Parse errors: 0

## night_drive_rain_electronic_001

- original_top: `<missing>`
- original_order: ``

### night_drive_rain_electronic_001_cf_ablation

- variant_type: `preference_ablation`
- top_choice: `<missing>` (same)
- ranking_order: ``
- expected_effect: 电子/氛围偏好的优势应降低，推荐理由不应过度引用电子偏好。
- error: `dry_run`

### night_drive_rain_electronic_001_cf_constraint

- variant_type: `constraint_flip`
- top_choice: `<missing>` (same)
- ranking_order: ``
- expected_effect: 高能量 EDM 候选的分数应上升；原先因为不要太吵而被压低的理由不应继续出现。
- error: `dry_run`

### night_drive_rain_electronic_001_cf_context

- variant_type: `context_flip`
- top_choice: `<missing>` (same)
- ranking_order: ``
- expected_effect: Festival Pulse 应更有竞争力；夜晚雨天和孤独感不应继续主导理由。
- error: `dry_run`

## party_warmup_discovery_001

- original_top: `<missing>`
- original_order: ``

### party_warmup_discovery_001_cf_ablation

- variant_type: `preference_ablation`
- top_choice: `<missing>` (same)
- ranking_order: ``
- expected_effect: 小众探索不应成为推荐 Unknown Patio 的主要理由。
- error: `dry_run`

### party_warmup_discovery_001_cf_constraint

- variant_type: `constraint_flip`
- top_choice: `<missing>` (same)
- ranking_order: ``
- expected_effect: Sofa Demo 04 应上升；groove 升温的权重应降低。
- error: `dry_run`

### party_warmup_discovery_001_cf_context

- variant_type: `context_flip`
- top_choice: `<missing>` (same)
- ranking_order: ``
- expected_effect: Hard Reset 的分数应上升；预热和不尴尬理由不应继续压制它。
- error: `dry_run`

## study_focus_chinese_vocal_001

- original_top: `<missing>`
- original_order: ``

### study_focus_chinese_vocal_001_cf_ablation

- variant_type: `preference_ablation`
- top_choice: `<missing>` (same)
- ranking_order: ``
- expected_effect: 中文女声的解释权重应降低。
- error: `dry_run`

### study_focus_chinese_vocal_001_cf_constraint

- variant_type: `constraint_flip`
- top_choice: `<missing>` (same)
- ranking_order: ``
- expected_effect: White Noise Terminal 应上升；中文女声偏好应让位于完全无人声约束。
- error: `dry_run`

### study_focus_chinese_vocal_001_cf_context

- variant_type: `context_flip`
- top_choice: `<missing>` (same)
- ranking_order: ``
- expected_effect: Drama Hook 的排名应上升；低打扰学习理由不应继续主导。
- error: `dry_run`

## Aggregate

- top-choice change rate: 0/9 (0.0%)

Interpretation note: top-choice changes are only a coarse signal. A faithful model can keep the same top choice while changing scores or reasons, and an unfaithful model can change rankings for the wrong reason.
