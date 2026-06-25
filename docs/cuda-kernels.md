# Custom CUDA Kernels

For hand-written CUDA ops (custom losses, fused ops, sampling kernels, etc.).

## Layout
```
csrc/                      # .cu / .cpp / .cuh sources
crafter_rl/kernels/        # Python bindings — loads the compiled extension
```
Sources in `csrc/` compile into a `crafter_rl._C` extension, loaded lazily by
`crafter_rl.kernels.load_kernels()` (via `torch.utils.cpp_extension`). Importing
`crafter_rl` never triggers a build — compilation happens only on first `load_kernels()` call.

## ⚠️ Prerequisite: the CUDA Toolkit (`nvcc`)
This machine has the NVIDIA **driver + runtime** (PyTorch ships its own CUDA runtime), but
**not the CUDA Toolkit compiler `nvcc`** — which is required to build custom kernels. Before
writing kernels, install a toolkit whose major version matches the torch build (`torch.version.cuda`):

```bash
# check what torch expects
uv run python -c "import torch; print(torch.version.cuda)"
# then install the matching CUDA Toolkit (provides nvcc), e.g. via the NVIDIA apt repo
# or:  uv pip install nvidia-cuda-nvcc-cu13   # pip-provided nvcc matching cu13 wheels
```
Verify with `nvcc --version`. Build will fail with a clear error from `load_kernels()` until
`nvcc` is on `PATH`.

## Adding a kernel
1. Drop `my_op.cu` (+ any `.cuh`) in `csrc/` with a `PYBIND11_MODULE` / `TORCH_LIBRARY` binding.
2. Call `crafter_rl.kernels.load_kernels()` to JIT-compile and import it (cached after first build).
3. Add a smoke test comparing the kernel against a pure-PyTorch reference for correctness.
4. Once proven, follow [promotion-checklist.md](promotion-checklist.md).

> For ahead-of-time builds (instead of JIT), graduate to a `setup.py`/`CMakeLists.txt` with
> `CUDAExtension` — only worth it once kernels stabilize.
