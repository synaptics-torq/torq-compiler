### CI/CD pipelines

### Development process

Development of a new feature happens in a dedicated `wip/NAME` branch in `synaptics-torq/torq-compiler-dev`.

The merging process is as follows:

1. As soon as there is some code the developer pushes a branch to `wip/NAME`
2. When the code is buildable and testable the developer opens a *draft* pull request from `wip/NAME` to `main`
3. The test GitHub workflow performs a build of the branch and basic tests with C model, reporting the results.
   The author can append the labels `test-aws` and `test-astra-machina-sl` to enable testing on the corresponding hardware.
4. The author fixes any issue identified at this point and re-pushes the branch until the tests pass
5. When the tests pass the author marks the PR as ready for review
6. At this point the test GitHub workflow performs a full test suite (including FPGA and real hardware).
  - In case of failures, the developer switches the PR back to draft, uses the labels mentioned above to 
    test on the problematic hardware, and fixes any remaining issues. Once the problematic tests pass, the 
    developer marks the PR as ready again.
7. The reviewers check the code, approve the PR, and submit it to the merge queue.
  - Maintainers can trigger a workflow to self-approve a PR by adding the label `maintainer-approved`; they
    can also skip the merge queue by forcing a merge to the base branch.
8. The test GitHub workflow rebases and squashes the commit on top of current main (or the latest merge queue
   branch) and re-tests the commit. If the tests fail, the PR is dequeued and the author needs to rebase and fix it.

### Release process

A Torq compiler release consists of the following artifacts:

- sources
- release tarball with:
  - torq-compiler for x86_64
  - torq-runtime for x86_64 (C model simulation)
  - torq-runtime for arm64 compatible with Astra SL SDK
  - torq Linux kernel module compatible with Astra SL SDK
- torq-compiler python wheel for x86_64
- torq-runtime python wheel for x86_64 and arm64 (Astra SL SDK)
- torq compiler docker image for x86_64
- user documentation
- compiler developer documentation
- docker image to build the compiler

Additionally, for CI purposes we publish the following artifacts:

- docker builder image used to build the compiler on AWS
- docker test image used to test the delivery on AWS
- docker fpga-test image used to test the delivery on AWS with FPGA instances

The release is performed on the following channels:

- GitHub repository: sources
- GitHub packages: docker images
- GitHub release assets: all other assets
- GitHub pages: documentation
- AWS ECR: AWS related images

The release process revolves around the following steps:

1. Developers publish sources to source repository in GitHub
2. GitHub CI creates the other deliverables and publishes them on GitHub releases and the package repository

The releases are performed on multiple parallel streams that correspond to git branches in the source repository:

- `main` : the main development branch where all new code is committed
- `vX.Y`: branch that tracks development for a given release stream

Each branch produces a snapshot release each time it is updated; older snapshot releases 
for the same branch are discarded.

The current snapshot releases are tracked with tags:

- `snapshot` : current release for the `main` branch
- `snapshot-vX.Y` : current snapshot release for the `vX.Y` branch

In addition to snapshot branches, there are also immutable releases `vX.Y.Z` on the corresponding 
version branch; these releases are tracked by tags `vX.Y.Z`.

Development of the compiler happens on the private `synaptics-torq/torq-compiler-dev` repository. Each
time a new commit is merged into the main or version branches, a private snapshot release is performed and
published in the repository.

Periodically the branches of this repository are pushed to the public `synaptics-torq/torq-compiler` 
repository. When this happens, the GitHub release workflow runs in the public repository, which 
builds and publishes the corresponding snapshot release.

Periodically, an official source release is tagged on `synaptics-torq/torq-compiler`, which 
triggers the GitHub release workflow that publishes the corresponding build artifacts.
