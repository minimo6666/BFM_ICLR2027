# Official SEDD binary-latent LSUN Churches baseline

This experiment keeps the official SEDD core unchanged:

- `model/transformer.py::SEDD` (small DDiT: 768, 12 blocks, 12 heads)
- `losses.py` Score Entropy objective and optimization manager
- `graph_lib.py::Uniform`
- `noise_lib.py::GeometricNoise`
- `sampling.py::AnalyticPredictor` and final `Denoiser`
- official optimizer, warmup, clipping, and EMA settings

Only the state space and data interface are specialized:

- vocabulary `50257 -> 2`
- input is the same frozen Churches BAE latent `[B,64,16,16]`
- spatial-major reversible serialization `[B,16,16,64] -> [B,16384]`
- public diffusion coordinate length `16384`, packed into `model.length=1024`
- each DDiT token parameterizes 16 binary coordinates; `scale_by_sigma=False`

Training is aligned with the DFM/BFM effective batch:

`8 per GPU * 2 GPUs * accumulation 12 = 192 examples/update`.

The BAE encoder remains frozen and is used to convert each image to binary
latents before the SEDD loss. The packed model keeps the official DDiT width,
depth, heads, Score Entropy loss, graph, noise schedule, and sampler interface,
while reducing the attention sequence from 16384 to 1024.

Sampling is exactly 64 NFE: 63 official analytic predictor calls plus the
official final denoiser call. At 10k--40k, 10,000 images are generated; at 50k,
50,000 images are generated. FID uses `metrics/fid_compute_algorithm_1.py`.

Run:

```bash
conda activate sedd
bash run_smoke_gpu2.sh
screen -dmS lsun_church_binary_sedd_gpu23 bash run_pipeline_gpu23.sh
```

