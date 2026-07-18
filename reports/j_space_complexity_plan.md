# J-Space Complexity Plan

This project now has two complementary probe sets:

1. Simple contrast probes:
   - `data/mechanistic_pilot_specs.jsonl`
   - Tests direct context/constraint flips.

2. Workspace-pressure probes:
   - `data/j_space_complexity_tasks.jsonl`
   - `runs/j_space_probe_specs.jsonl`
   - Tests routine tasks against complex planning/conflict/repair tasks.

## Why This Matters

If the workspace/J-space signal is weak for routine tasks, then a simple
music-rerank experiment may produce a false negative. The task may be too easy,
not the idea wrong.

## First Comparison

Run all three routine/complex pairs:

| Family | Routine | Complex | Added Complex Signals |
|---|---|---|---|
| study_focus | direct no-vocal pick | tool-conflict repair | constraint conflict, evidence integration, uncertainty, repair |
| party_energy | direct rave pick | feedback repair | goal maintenance, constraint balance, repair |
| night_drive | single-step rerank | sequence planning | sequence planning, novelty, constraint balance |

## Expected Result

The complex side should show stronger internal movement around:

- `constraint_conflict`
- `evidence_integration`
- `uncertainty`
- `repair`
- `goal_maintenance`
- `sequence_planning`

The routine side may mostly activate item/context features such as:

- `instrumental_focus`
- `rave_hard_energy`
- `rain_night_drive`

## Next Step

Extend `run_sae_probe.py` or a J-lens notebook to accept
`runs/j_space_probe_specs.jsonl` and compare routine text against complex text.

