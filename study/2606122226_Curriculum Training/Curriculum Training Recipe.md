# Curriculum Training Recipe

## Global Rule

Use one continuous LR schedule over all optimizer steps in each stage.
Do not reset LR at epoch boundaries.
Do not decay LR to zero once per epoch.

## Stage 1: Original Data & Proven 0.86 training config

Data:
- Original data only.

Epochs:
- `1` epoch.

LR:
- Peak LR: `2e-4`

Schedule:
- Decay: linear decay to `0`.

Goal:
- Fit the output format.
- Fit the easy categories.
- Teach the answer template for difficult categories, without expecting real difficult-category skill yet.

Note:
- No soup saving is needed. Resummable checkpoint still required.
- Submit stage 1 submission.zip after stage 1 training.

## Stage 2: First Difficult-Category Learning

Data:
1. 100% original data replay
2. Aug1 data: 50% of the total aug data, stratified sampling with bias on short CoT. Must cover all the types/scenarios/etc. in each category of puzzle, just sample more of the shorter samples and fewer long samples.

Epochs:
- `1` epoch.

LR:
- Peak LR: `1.5e-4`

Schedule (WSD):
- Warmup: first `5%` of total optimizer steps.
- Stable: hold peak LR until `70%` of total optimizer steps.
- Decay: linear decay to `0` over the final `30%` of steps.

Goal:
- Perform the first real learning of difficult-category CoT behavior.
- Introduce long CoT gradually.
- Preserve original/easy-category behavior through replay.

Note:
- No soup saving is needed. Resummable checkpoint still required.
- Submit stage 2 submission.zip after stage 2 training.


## Stage 3: Full Augmented Data

Data:
1. all original data
2. all aug1 data
3. all aug2 data

Full augmented pool:
- Include all Stage 2 augmented examples.
- Include all remaining augmented examples.
- Include all short, medium, and long CoT examples.
- Keep category / operation / template proportions balanced.

Epochs:
- `1` epoch

LR:
- Peak LR: `1.5e-4`

Schedule (WSD):
- Compute total optimizer steps across both epochs.
- Warmup: first `5%` of total optimizer steps.
- Stable: hold peak LR until `70%` of total optimizer steps.
- Decay: linear decay to `0` over the final `30%` of steps.

Goal:
- Perform full adaptation to the complete augmented difficult-category distribution.
- Deepen and stabilize difficult-category CoT skill on the full augmented distribution.
- Keep original/easy-category behavior anchored with replay.

Note:
- Soup lora saving is needed. Resummable checkpoint still required. We will decide if we build soup after we see the result.
- Submit stage 3 submission.zip after stage 3 training.
