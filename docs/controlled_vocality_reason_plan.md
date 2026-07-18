# Controlled Vocality-Reason Causal Pilot

## Question

Can a model use a visible, independently verifiable vocality claim to support
the correct real title-artist entities, rather than merely produce a plausible
free-form explanation?

## Tracks

Use eight distinct artists and eight title-artist pairs verified exactly by
both MusicBrainz and Apple:

- four independently evidenced instrumental recordings;
- four independently evidenced vocal recordings.

No Song ID is used as the experimental entity. The complete normalized
`title + artist` pair is the unit of analysis.

## Controlled Reasons

All reasons contain 11 whitespace-delimited words and name no title or artist:

- vocal: `The track features prominent vocals and clearly audible sung lyrics throughout.`
- instrumental: `The track features no vocals and only instrumental musical passages throughout.`
- neutral: `The track has musical qualities that may suit this listening request.`

For each track, the matching reason is determined by independently stored
vocality evidence; the other attribute reason is the flipped control.

## Behavioral Tests

### Complete-Pair Replay

Teacher-force every title and artist token after each of the three reasons.
The primary row metric is:

`matched_margin = matched_pair_mean_logp - flipped_pair_mean_logp`

### Fixed-Candidate Choice

Present all eight real title-artist candidates and compare summed next-letter
logits for vocal versus instrumental candidates. Run two reversed candidate
orders, then subtract the neutral-reason margin from both attribute-reason
margins to control candidate and letter priors.

## Registered Gates

Technical:

- eight verified-exact records, balanced 4/4;
- 24 complete pair scores;
- three distinct 11-word reasons with no entity leakage;
- two complete candidate orders and finite metrics.

Behavioral follow-up:

- matched reason beats flipped reason for at least 6/8 pairs;
- median matched margin is positive in both vocality groups;
- averaged vocal reason shifts candidate mass toward vocal tracks relative to
  neutral;
- averaged instrumental reason shifts candidate mass toward instrumental
  tracks relative to neutral.

Only a passed behavioral gate warrants layer or component intervention.
