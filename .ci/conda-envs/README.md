# conda-envs

This directory contains files used to create `conda` environments for development
and testing of LightGBM.

The `.txt` files here are intended to be used with `conda create --file`.

For details on that, see the `conda` docs:

* `conda create` docs ([link](https://conda.io/projects/conda/en/latest/commands/create.html))
* "Managing Environments" ([link](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))

The ordinary CI environment requires `setuptools>=78.1.1` to include the fixes for
[CVE-2024-6345](https://github.com/advisories/GHSA-cx63-2mw6-8hw5) and
[CVE-2025-47273](https://github.com/advisories/GHSA-5rjg-fvgr-3xxf).

The separate `pixi` environment `py310` tests older supported dependencies.
Its pinned pandas 1.3 package requires `setuptools<60`, so its lockfile deliberately
retains an affected version. Use it only for compatibility tests on disposable,
unprivileged runners without credentials, private data, or writable host mounts.
Do not use it to fetch untrusted package URLs or for production/release builds;
use `ci-core.txt` or the modern default `pixi` environment for those purposes.
