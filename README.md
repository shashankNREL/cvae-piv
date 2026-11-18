# CVAE-PIV: Gap-Filling of PIV Velocity Fields

Code accompanying the paper (currently under review). This code uses a conditional variational autoencoder (CVAE) built from partial convolutions to reconstruct missing regions in particle image velocimetry (PIV) velocity fields.

## Method overview

- **Architecture:** a partial-convolution CVAE encoder/decoder with skip connections. Each snapshot is conditioned on labels (time index, per-component min/max and field sums).
- **Loss:** masked MSE reconstruction + β·(KLD − target)² + γ·vorticity MSE.
- **3-phase schedule:**
  1. Reconstruction only (β = 0, γ = 0)
  2. Cyclical β annealing
  3. β frozen, γ ramps up linearly
- **Anti-collapse measures:** a target-KLD loss, dropout on skip connections, and 1×1 bottleneck convolutions on skip connections.

## Requirements

- Python 3, PyTorch, NumPy, scikit-learn, joblib, matplotlib
- [`partialconv`](https://github.com/NVIDIA/partialconv), which must be importable as `partialconv.models.partialconv2d`
- Optional: Ray Tune, for hyperparameter search

## Data

The input is an `.npz` file holding the gapped fields, with `ImgScale` (normalized velocity, 3 components) and `Mask`. When you train with holes (`-H 1`, the default), the script also looks for a matching clean file. It finds that file by replacing `Holes` with `Clean` in the filename. The clean file must contain `ImgScale`, `X`, `Y`, `Min` and `Max`.

## Usage

**Train and evaluate:**
```bash
python piv_gapfil_phased_loss.py -f PIV_Holes.npz -e 500 -b 64 -l 16 -lr 1e-4 --seed 0
```
The results go to a time-stamped `*_phased/` directory, which holds the model checkpoint, the label scaler and `AllImagesResults_phased_loss.npz`.

**Hyperparameter search (Ray Tune):**
```bash
python piv_gapfil_phased_loss.py -f PIV_Holes.npz --tune --tune_samples 10
```

**Turbulence-statistics post-processing (no training):**
```bash
python piv_gapfil_phased_loss.py --postprocess \
    --clean-file PIV_Clean.npz --prediction-file AllImagesResults_phased_loss.npz \
    --holes-file PIV_Holes.npz --model-file <checkpoint> \
    --postprocess-output-dir postprocess_statistics
```
This step produces summary statistics (CSV/JSON), PDFs of the fields, structure functions, error histograms and snapshot plots.

Run `python piv_gapfil_phased_loss.py -h` to see every option, including `--max_beta`, `--target_kld`, `--max_gamma`, `--skip_dropout`, `--n_cycles`, `--phase1_frac` and `--phase2_frac`.

## Citation

Citation details will be added once the paper is published.
