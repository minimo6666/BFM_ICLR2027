# Codex integration checklist

Copy files into the BFM_ICLR2027 repo preserving paths.

Do not edit:
- current `binarylatent_flow_expectation_consistent_retrain.py`
- corrected posterior math
- sampling grid
- hard final
- old BLD files

Run syntax checks:

```bash
python -m py_compile models/binarylatent_flow_controlled_src.py
python -m py_compile experiments/ICLR27/src_controlled/train/train_controlled_src.py
python experiments/ICLR27/src_controlled/sanity/inspect_one_step_src.py
```

Primary controlled matrix:

1. `original`
   - current BFM
   - train t / sample t
   - base loss

2. `aligned_bce`
   - train t-1 / sample t-1
   - base loss

3. `aligned_brier`
   - exactly same sampler as aligned_bce
   - exactly same one-forward training state
   - base + unweighted Brier

4. `aligned_src`
   - exactly same sampler as aligned_bce
   - exactly same one-forward training state
   - base + normalized one-step S^2 weighted Brier

Use same training arguments and seed.

For the first report, evaluate all four at 64 NFE with the same FID pipeline.
Do not introduce an adaptive sampler or low-NFE-matched training yet.
