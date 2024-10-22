# nuclai

[![License BSD-3](https://img.shields.io/pypi/l/nuclai.svg?color=green)](https://github.com/dsethz/nuclai/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/nuclai.svg?color=green)](https://pypi.org/project/nuclai)
[![Python Version](https://img.shields.io/pypi/pyversions/nuclai.svg?color=green)](https://python.org)
[![tests](https://github.com/dsethz/nuclai/workflows/tests/badge.svg)](https://github.com/dsethz/nuclai/actions)
[![codecov](https://codecov.io/gh/dsethz/nuclai/branch/main/graph/badge.svg)](https://codecov.io/gh/dsethz/nuclai)

A software to identify the presence of cell surface markers based on nuclear signal.

----------------------------------

This repository was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

## Installation
If you do not have python installed already, we recommend installing it using the
[Anaconda distribution](https://www.anaconda.com/products/distribution). Installing `nuclai` takes ~5 min and
was tested with `python 3.12.7`.

### Virtual environment setup
If you do not use and IDE that handles [virtual environments](https://realpython.com/python-virtual-environments-a-primer/)
for you (e.g. [PyCharm](https://www.jetbrains.com/pycharm/)) use your command line application (e.g. `Terminal`) and
one of the many virtual environment tools (see [here](https://testdriven.io/blog/python-environments/)). We will
use `conda`

1) Create new virtual environment

    ```bash
    conda create -n nuclai python=3.12.7
    ```

2) Activate virtual environment

    ```bash
    conda activate nuclai
    ```

### pip installation
Recommended if you do not want to develop the `nuclai` code base.

3) Install `nuclai`
    ```bash
    # update pip
    pip install -U pip==23.2.1
    pip install nuclai
    ```

4) (Optional) `GPUs` greatly speed up training and inference of `nuclai` and are available for
`Windows` and `Linux`. Check if your `GPU(s)` are CUDA compatible
([`Windows`](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#verify-you-have-a-cuda-capable-gpu),
 [`Linux`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#verify-you-have-a-cuda-capable-gpu)) and
 update their drivers if necessary.

5) [Install `torch`/`torchvision`](https://pytorch.org/get-started/previous-versions/) compatible with your system.
`nuclai` was tested with `torch` version `2.4.1`, `torchvision` version `0.19.1`, and `cuda` version
`12.1.1`. Depending on your OS, your `CPU` or `GPU` (and `CUDA` version) the installation may change

```bash
# Windows/Linux CPU
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

# Windows/Linux GPU (CUDA 11.3.X)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# macOS CPU
pip install torch==2.4.1 torchvision==0.19.1

```

6) [Install `lightning`](https://lightning.ai/pytorch-lightning). `nuclai` was tested with version `2.4.0`.

```bash
pip install lightning==2.4.0
```


### Source installation
Installation requires a command line application (e.g. `Terminal`) with
[`git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [python](https://www.python.org) installed.
If you operate on `Windows` we recommend using
[`Ubuntu on Windows`](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview).
Alternatively, you can install [`Anaconda`](https://docs.anaconda.com/anaconda/user-guide/getting-started/) and
use `Anaconda Powershell Prompt`. An introductory tutorial on how to use `git` and GitHub can be found
[here](https://www.w3schools.com/git/default.asp?remote=github).

3) (Optional) If you use `Anaconda Powershell Prompt`, install `git` through `conda`

    ```bash
    conda install -c anaconda git
    ```

4) clone the repository (consider `ssh` alternative)

    ```bash
    # change directory
    cd /path/to/directory/to/clone/repository/to

    git clone https://github.com/dsethz/nuclai.git
    ```

5) Navigate to the cloned directory

    ```bash
    cd nuclai
    ```

6) Install `nuclai`
    ```bash
    # update pip
    pip install -U pip
    ```

    1) as a user

        ```bash
        pip install .
        ```
    2) as a developer (in editable mode with development dependencies and pre-commit hooks)

        ```bash
        pip install -e ".[testing]"
        pre-commit install
        ```

7) (Optional) `GPUs` greatly speed up training and inference of `nuclai` and are available for
`Windows` and `Linux`. Check if your `GPU(s)` are CUDA compatible
([`Windows`](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#verify-you-have-a-cuda-capable-gpu),
 [`Linux`](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#verify-you-have-a-cuda-capable-gpu)) and
 update their drivers if necessary.

8) [Install `torch`/`torchvision`](https://pytorch.org/get-started/previous-versions/) compatible with your system.
`nuclai` was tested with `torch` version `2.4.1`, `torchvision` version `0.19.1`, and `cuda` version
`12.1.1`. Depending on your OS, your `CPU` or `GPU` (and `CUDA` version) the installation may change

```bash
# Windows/Linux CPU
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

# Windows/Linux GPU (CUDA 11.3.X)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# macOS CPU
pip install torch==2.4.1 torchvision==0.19.1

```

9) [Install `lightning`](https://lightning.ai/pytorch-lightning). `nuclai` was tested with version `2.4.0`.

```bash
pip install lightning==2.4.0
```


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"nuclai" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/dsethz/nuclai/issues

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
