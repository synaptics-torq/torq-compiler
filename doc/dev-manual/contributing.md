# Recommended submission process

To contribute to this repository the recommended process is the following:

1. Create a work-in-progress branch locally. The branch name should start with `wip/` so that
   it can later be pushed to GitHub:

   ```{code} shell
   $ git checkout -b wip/mytopic
   ```

2. Perform your changes locally and compile the project

3. Check that all the required test cases pass as described in [Run tests](testing_howto.md#run-tests):

   ```{code} shell
   $ cd torq-compiler
   $ pytest -m ci
   ```

4. Reformat the code to match the coding style:

   ```{code} shell
   $ cd torq-compiler
   $ scripts/format-code
   ```

5. Commit changes locally and push branch to GitHub:

   ```{code} shell
   $ cd torq-compiler
   $ git add . 
   $ git commit
   $ git push --set-upstream origin wip/mytopic
   ```

   You can push multiple times the branch. You can both amend the current
   commit or create multiple commits depending on your needs.

   :::{note}
   You can always push to GitHub changes that are not yet ready for review as 
   soon as you want. The wip branches are not checked by CI nor submitted for 
   review until you create a pull request for them.
   :::

6. When the change is ready for review you can create a pull request on GitHub.
   Write its title and description as described in
   [Pull Request Description and Commit Message](#pull-request-description-and-commit-message).

7. Check that all the tests for the PR on CI pass, address any problem by
   pushing the fix to the branch (either as an amend or a new commit).

   By default PRs are not tested on FPGA. If you want to test the PR on the AWS
   FPGA add the ``test-aws`` label, if you want to test on the SOC FPGA add the 
   ``test-soc`` label. For security reasons before tests are run on the 
   corresponding FPGA a maintainer must review and approve the deployment 
   (maintainers can self-approve their own requests).

8. Address any change discussed during review. It is normally better to commit
   these changes as a new commit instead of performing an amend of the original
   commit to facilitate review.

9. Once the changes are approved they will be squashed and rebased on top of
   current main.


## Pull Request Description and Commit Message

PRs are squashed into a single commit on merge (step 9), so write the PR title and
description as that commit's message. When a PR has a single commit, give that commit the
same message.

**Title**: `[Area] Summary in the imperative`. Area is one of `Compiler`, `Runtime`,
`TorqHW`, `Testing`, `CI`, `Deps`, `Doc`, e.g.
`[Compiler] Keep both inputs of elementwise compares dense`.

**Body**: four sections, in this order:

- **Problem**: what is wrong or missing, as seen by a user or a test: error message,
  wrong results, affected models or shapes, issue numbers.
- **Context**: why it happens and what the reader needs to understand the fix: the code
  path involved, earlier commits that led here (by short sha and title), constraints.
- **Solution**: the approach, in a few sentences.
- **Changes**: one bullet per change (pass, op, file group, tests, xfails, docs), with the
  reason for any non-obvious choice. End with a **Results** bullet listing what was run
  and on what (tests, chips, board numbers, the `main` sha compared against).

In the PR description use `## Problem`, `## Context`, `## Solution`, `## Changes`
headings. In a commit message use `Problem:`, `Context:`, `Solution:`, `Changes:` lines,
and wrap lines at about 100 characters.
GitHub fills the description of a single-commit PR from its commit message, so change those
lines to `##` headings when you open the PR.

Put `Fixes #<issue>` on its own line after the sections for each issue the PR closes.

Example (trimmed from 3841f1b36):

```text
[Deps] Pick the int16 strided transpose_conv2d fix in llvm-project

Problem:
int16 transpose convolutions with a stride above 1 fail to compile, with both torq-compile and
the llvm-cpu reference: "'tosa.pad' op expect input and output to have same element type, got
'i16' and 'i8'" (#2845). int8 versions of the same layers compile.

Context:
- The strided decomposition (TosaDecomposeTransposeConv.cpp in llvm-project) pads the weights
  with a pad constant built from the input type, so an i8 weight tensor gets an i16 pad value.
- Upstream already fixed it: llvm/llvm-project@1a2f7bc8a966.

Solution:
Cherry-pick the upstream fix into synaptics-torq/llvm-project, bump it in synaptics-torq/iree,
and bump third_party/iree here.

Changes:
- third_party/iree: da5a3c63, which bumps llvm-project to the cherry-pick plus the upstream test.
- test_keras_ops-xfails.txt: removes the 104 entries of #2845.
- Results (SL2610 and SR200 simulators, keras transpose tests): 227 passed, 30 xpassed,
  3 xfailed, against 223 passed, 33 xfailed and 4 errors on main.

Fixes #2845
```

## Coding Style

The coding style used reflects as much as possible the conventions used in the IREE codebase,
with just a few minor deviations.

A few important conventions:

- Indentation is 4 spaces (no tabs)
- All identifiers are _CamelCase_
- Types start with an Uppercase letter while variables, functions and methods with lowercase.
  Only preprocessor macros are completely UPPER_CASE.
- Column width is limited to 100 characters
- No trailing empty spaces in lines

:::{important}
Use the automatic formatter script before submitting code:
```{code} shell
$ torq-compiler/scripts/format-code
```
The formatter will change the files in place, make sure you commit these changes.
:::

## Updating this Documentation

This documentation can easily be modified by editing the ``.md`` files in the
[doc](https://github.com/synaptics-torq/torq-compiler/tree/main/doc)
folder. Follow the instructions below to render the documents with or without
docker, and then serve and open them in a browser.

- **Render locally with Docker**

   Run `torq-compiler/doc/build.sh`.


- **Render locally without Docker**

   1. clone the [synaptics-sphinx-theme](https://github.com/syna-astra-dev/synaptics-sphinx-theme) 
      repository

      ```{code} shell
      $ git clone https://github.com/syna-astra-dev/synaptics-sphinx-theme.git
      ```

   2. install the theme from the cloned repository `pip install -e synaptics-sphinx-theme`
      (note the `-e`, without it Sphinx will not be able to load the theme);
   3. render the documents `sphinx-build torq-compiler/doc torq-compiler/doc/_build/html`;

- **Serve and open the rendered documents**

   1. serve `torq-compiler/doc/_build/html` (e.g.
      `python -m http.server -d torq-compiler/doc/_build/html 8080`); and
   2. open [localhost:8080](http://localhost:8080) in a browser.
