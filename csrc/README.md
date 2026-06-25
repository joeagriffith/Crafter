# csrc/

C++/CUDA sources for custom ops. Drop `.cu` / `.cpp` files here; they are
JIT-compiled and loaded as the `crafter_rl._C` extension module on demand via:

```python
from crafter_rl.kernels import load_kernels
_C = load_kernels()   # torch.utils.cpp_extension.load(name="crafter_rl_C", sources=glob("csrc/*.cu", "csrc/*.cpp"))
```

Importing `crafter_rl` (or `crafter_rl.kernels`) never triggers a build --
compilation only happens when you explicitly call `load_kernels()`. With no
sources present it raises a clear `NotImplementedError`.

## Requirements

Building requires the **CUDA Toolkit (`nvcc`)** to be installed and on `PATH`,
matching the PyTorch CUDA version. **It is NOT currently installed** in this
environment -- install it first (e.g. via the NVIDIA CUDA Toolkit) before
adding sources and calling `load_kernels()`.
