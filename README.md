pyFRESEAN
==============================
[//]: # (Badges)

| **Latest release** | [![Last release tag][badge_release]][url_latest_release] ![GitHub commits since latest release (by date) for a branch][badge_commits_since]  [![Documentation Status][badge_docs]][url_docs]|
| :----------------- | :------- |
| **Status**         | [![GH Actions Status][badge_actions]][url_actions] [![codecov][badge_codecov]][url_codecov] |
| **Community**      | [![License: MIT License][badge_license]][url_license]  [![Powered by MDAnalysis][badge_mda]][url_mda]|

[badge_actions]: https://github.com/HeydenLabASU-collab/pyFRESEAN/actions/workflows/gh-ci.yaml/badge.svg
[badge_codecov]: https://codecov.io/gh/HeydenLabASU-collab/pyFRESEAN/branch/main/graph/badge.svg
[badge_commits_since]: https://img.shields.io/github/commits-since/HeydenLabASU-collab/pyFRESEAN/latest
[badge_docs]: https://readthedocs.org/projects/pyfresean/badge/?version=latest
[badge_license]: https://img.shields.io/badge/License-MIT-yellow.svg
[badge_mda]: https://img.shields.io/badge/powered%20by-MDAnalysis-orange.svg?logoWidth=16&logo=data:image/x-icon;base64,AAABAAEAEBAAAAEAIAAoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJD+XwCY/fEAkf3uAJf97wGT/a+HfHaoiIWE7n9/f+6Hh4fvgICAjwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACT/yYAlP//AJ///wCg//8JjvOchXly1oaGhv+Ghob/j4+P/39/f3IAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJH8aQCY/8wAkv2kfY+elJ6al/yVlZX7iIiI8H9/f7h/f38UAAAAAAAAAAAAAAAAAAAAAAAAAAB/f38egYF/noqAebF8gYaagnx3oFpUUtZpaWr/WFhY8zo6OmT///8BAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgICAn46Ojv+Hh4b/jouJ/4iGhfcAAADnAAAA/wAAAP8AAADIAAAAAwCj/zIAnf2VAJD/PAAAAAAAAAAAAAAAAICAgNGHh4f/gICA/4SEhP+Xl5f/AwMD/wAAAP8AAAD/AAAA/wAAAB8Aov9/ALr//wCS/Z0AAAAAAAAAAAAAAACBgYGOjo6O/4mJif+Pj4//iYmJ/wAAAOAAAAD+AAAA/wAAAP8AAABhAP7+FgCi/38Axf4fAAAAAAAAAAAAAAAAiIiID4GBgYKCgoKogoB+fYSEgZhgYGDZXl5e/m9vb/9ISEjpEBAQxw8AAFQAAAAAAAAANQAAADcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAjo6Mb5iYmP+cnJz/jY2N95CQkO4pKSn/AAAA7gAAAP0AAAD7AAAAhgAAAAEAAAAAAAAAAACL/gsAkv2uAJX/QQAAAAB9fX3egoKC/4CAgP+NjY3/c3Nz+wAAAP8AAAD/AAAA/wAAAPUAAAAcAAAAAAAAAAAAnP4NAJL9rgCR/0YAAAAAfX19w4ODg/98fHz/i4uL/4qKivwAAAD/AAAA/wAAAP8AAAD1AAAAGwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALGxsVyqqqr/mpqa/6mpqf9KSUn/AAAA5QAAAPkAAAD5AAAAhQAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADkUFBSuZ2dn/3V1df8uLi7bAAAATgBGfyQAAAA2AAAAMwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB0AAADoAAAA/wAAAP8AAAD/AAAAWgC3/2AAnv3eAJ/+dgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA9AAAA/wAAAP8AAAD/AAAA/wAKDzEAnP3WAKn//wCS/OgAf/8MAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIQAAANwAAADtAAAA7QAAAMAAABUMAJn9gwCe/e0Aj/2LAP//AQAAAAAAAAAA
[badge_release]: https://img.shields.io/github/release-pre/HeydenLabASU-collab/pyFRESEAN.svg
[url_actions]: https://github.com/HeydenLabASU-collab/pyFRESEAN/actions?query=branch%3Amain+workflow%3Agh-ci
[url_codecov]: https://codecov.io/gh/HeydenLabASU-collab/pyFRESEAN/branch/main
[url_docs]: https://pyfresean.readthedocs.io/en/latest/?badge=latest
[url_latest_release]: https://github.com/HeydenLabASU-collab/pyFRESEAN/releases
[url_license]: https://opensource.org/licenses/MIT
[url_mda]: https://www.mdanalysis.org

A Python package for FRESEAN based analysis

pyFRESEAN is bound by a [Code of Conduct](https://github.com/HeydenLabASU-collab/pyFRESEAN/blob/main/CODE_OF_CONDUCT.md).

### Quick start

```python
import MDAnalysis as mda
from pyfresean import FRESEAN, Align, Unwrap

u = mda.Universe("topol.tpr", "traj.trr")
sel = u.select_atoms("all")
u_ref = mda.Universe("min.xyz")

# optional preprocessing (standard MDA trajectory transforms)
u.trajectory.add_transformations(
    Unwrap(u.atoms),
    Align(
        sel,
        reference_positions=u_ref.atoms.positions,
        place_com_in_box=False,  # gas phase
    ),
)

analysis = FRESEAN(
    u,
    select="all",
    n_constraints=6,
    n_corr=500,
    dt=0.004,
    sigma=10.0,
)
analysis.run()

print(analysis.results.freqs)
print(analysis.results.eigenvalues)
print(analysis.results.vdos_total)
```

See also the minimal notebooks in [`examples/`](examples/).

### Tests

```bash
pip install -e ".[test]"
pytest                    # unit tests (default)
pytest -m pytutorial_ref   # vs Python tutorial reference (~90 s; may download trajectory)
pytest -m c_ref           # vs FRESEAN COARSE (C) reference
```

Reference data:
- `pyfresean/tests/data/fresean_pytutorial_ref/` — Python tutorial comparisons
- `pyfresean/tests/data/fresean_c_ref/` — FRESEAN COARSE (C) comparisons

On CI, every pull request runs unit tests plus all `pytutorial_ref` and
`c_ref` regressions. Download the **`c-ref-plots-*`** artifact from the
workflow run (Actions tab) for eigenvalue and VDoS PNG comparisons.

### Installation

To build pyFRESEAN from source,
we highly recommend using virtual environments.
If possible, we strongly recommend that you use
[Anaconda](https://docs.conda.io/en/latest/) as your package manager.
Below we provide instructions both for `conda` and
for `pip`.

#### With conda

Ensure that you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

Create a virtual environment and activate it:

```
conda create --name pyfresean
conda activate pyfresean
```

Install the development and documentation dependencies:

```
conda env update --name pyfresean --file devtools/conda-envs/test_env.yaml
conda env update --name pyfresean --file docs/requirements.yaml
```

Build this package from source:

```
pip install -e .
```

If you want to update your dependencies (which can be risky!), run:

```
conda update --all
```

And when you are finished, you can exit the virtual environment with:

```
conda deactivate
```

#### With pip

To build the package from source, run:

```
pip install .
```

If you want to create a development environment, install
the dependencies required for tests and docs with:

```
pip install ".[test,doc]"
```

### Copyright

The pyFRESEAN source code is hosted at https://github.com/HeydenLabASU-collab/pyFRESEAN
and is available under the [MIT License](https://opensource.org/licenses/MIT) (see the file [LICENSE](https://github.com/HeydenLabASU-collab/pyFRESEAN/blob/main/LICENSE)).

Copyright (c) 2026, Amruthesh Thirumalaiswamy


#### Acknowledgements
 
Project based on the 
[MDAnalysis Cookiecutter](https://github.com/MDAnalysis/cookiecutter-mda) version 0.1.
Please cite [MDAnalysis](https://github.com/MDAnalysis/mdanalysis#citation) when using pyFRESEAN in published work.
