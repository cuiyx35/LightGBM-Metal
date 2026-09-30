Helper Scripts for CI
=====================

This folder contains scripts which are run on CI services.

Dockerfile used on CI service is maintained in a separate [GitHub repository](https://github.com/guolinke/lightgbm-ci-docker) and can be pulled from [Docker Hub](https://hub.docker.com/r/lightgbm/vsts-agent).

# CI platform checks

Ordinary Python jobs test the installed CPU package and general examples.
The explicit inventory in `metal-examples.txt` is exercised by `metal_macos.yml`
with this checkout's Metal-enabled native library on Apple Silicon: both
validation scripts and all three benchmark smoke workloads. The inventory must
be updated together with those workflows when adding another Metal example.

Source distributions are checked against the isolated sources prepared by
`build-python.sh`. `check-sdist.py` requires the exact file names and bytes,
plus the backend-generated `PKG-INFO`; it rejects missing, duplicate, unexpected,
and unsafe entries. `pydistcheck` still checks sizes and distribution properties.
This allows reviewed source additions without a manually maintained file count.

## PowerPC

GitHub does not provide a standard `ubuntu-24.04-ppc64le` hosted runner.
The C++ workflow instead uses a standard Ubuntu runner with Docker's official
QEMU setup action to compile and execute the complete Debug/OpenMP C++ test
target in an Ubuntu `linux/ppc64le` container. It checks the target architecture
and does not filter tests. The aggregate C++ check requires this job to pass.

The action commit, emulator image digest, and dated Ubuntu image digest are
pinned in `cpp.yml`; update them together with a successful complete test run.
QEMU's binfmt registration exists only on the disposable hosted runner. The
test container has no privileged flag and mounts the checkout read-only, then
builds in its own filesystem. No self-hosted runner or repository credential is
passed into the container.

This verifies PowerPC compilation and emulated functional behavior, not native
hardware performance, timing, or every hardware-specific OpenMP interaction.
It retains platform coverage without claiming equivalence to native testing.

The older dependency compatibility environment and its setuptools constraint
are documented in `conda-envs/README.md`.
