# PyFRESEAN examples

Minimal notebooks for FRESEAN analysis with the `pyfresean` package.

| Notebook | System | Tutorial |
|---|---|---|
| `01_MD-alanine-dipeptide-gas-50K.ipynb` | Alanine dipeptide, gas, 50 K | `01-tutorial_MD-gas-50K.ipynb` |
| `02_MD-alanine-dipeptide-gas-300K.ipynb` | Alanine dipeptide, gas, 300 K | `02-tutorial_MD-gas-300K.ipynb` |
| `03_MD-alanine-dipeptide-water-300K.ipynb` | Alanine dipeptide in water, 300 K | `03-tutorial_MD-water-300K.ipynb` |
| `04_CG-alanine-dipeptide-gas-300K.ipynb` | CG + FRESEAN on alanine dipeptide; CG vs all-atom VDoS comparison | `02-tutorial_MD-gas-300K.ipynb` (same trajectory) |
| `05_CG-hewl.ipynb` | FRESEAN-metadynamics workflow for HEWL (CG, modes 7 & 8, PLUMED prep) | [Zenodo 10.5281/zenodo.15678841](https://doi.org/10.5281/zenodo.15678841) |

## Setup

```bash
pip install -e ".[examples]"
```

Run notebooks from the `examples/` directory so `input_data/` and `output_data/` resolve correctly.

- **Inputs:** `input_data/` — see `input_data/README.md` for trajectory files.
- **Outputs:** `output_data/` — generated files; gitignored except the README.
- **Metadynamics templates:** `fresean_metaD_data/` — PLUMED/GROMACS files for notebook 05.

## What each notebook covers

MD notebooks (01–03):

1. total VDoS (0–200 cm⁻¹) with peak markers
2. 1D-VDoS of the top 4 modes at a selected peak (+ sum)
3. leading eigenvalues at each low-frequency peak (water: includes 0 cm⁻¹)
4. 5×5 mode time-correlation grid at a selected peak
5. weighted clustering matrix
6. clustered key-vibration VDoS (up to 7 modes + sum)
7. clustered-mode overlap matrix
8. water only: zero-frequency mode contributions

Notebook 04 adds coarse-graining, back-mapping, PLUMED mode PDB output, and an all-atom vs CG total VDoS comparison on the same frames.

Not included here (full tutorials only): harmonic-mode comparison, 3D mode animations.

VDoS line plots use cubic-spline interpolation (`pyfresean.postprocess.plot_spectra`), same display style as the FRESEAN_tutorial notebooks. The underlying FRESEAN data are unchanged.

The full tutorials add clustering, harmonic-mode comparison, and 3D visualizations.
