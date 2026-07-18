# J-Space Complexity Experiment Template

## Run Metadata

- Date:
- Model:
- Readout method:
  - SAE
  - J-lens
  - logit lens
  - behavioral only
- Layer / hook:
- Probe specs:
  - `runs/j_space_probe_specs.jsonl`
- Notes:

## Hypothesis

Routine music recommendation tasks should show weaker workspace-pressure
signals than complex tasks involving conflict, evidence integration, planning,
or repair.

## Probe Results

### Probe: `workspace_pressure__routine_quiet_study_pick__vs__complex_study_tool_conflict_repair`

- Routine task behavior:
- Complex task behavior:
- Added complex signals:
  - `constraint_conflict`
  - `evidence_integration`
  - `repair`
  - `uncertainty`
- Internal movement:
- Interpretation:

### Probe: `workspace_pressure__routine_party_energy_pick__vs__complex_party_feedback_repair`

- Routine task behavior:
- Complex task behavior:
- Added complex signals:
  - `constraint_balance`
  - `goal_maintenance`
  - `repair`
  - `party_warmup_groove`
- Internal movement:
- Interpretation:

### Probe: `workspace_pressure__routine_night_drive_pick__vs__complex_night_drive_playlist_planning`

- Routine task behavior:
- Complex task behavior:
- Added complex signals:
  - `constraint_balance`
  - `novelty_discovery`
  - `sequence_planning`
- Internal movement:
- Interpretation:

## Aggregate Notes

- Which complex signals moved clearly?
- Which signals did not appear?
- Did complex tasks show stronger feature/J-space movement than routine tasks?
- Did behavior improve without matching internal signal?
- Did internal signal move without clear behavioral difference?

## Claim Boundary

Write the claim conservatively:

> In this pilot, complex music recommendation tasks produced [stronger/weaker/no
> clearer] internal movement around [signals] than routine tag-matching tasks.

Do not claim direct J-space measurement unless the readout is actually
Jacobian-lens/J-space based.

