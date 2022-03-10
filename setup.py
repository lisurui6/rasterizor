from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []


ext_modules = [
    CUDAExtension('rasterizor.cuda.rasterize', [
        'rasterizor/cuda/rasterize_cuda.cpp',
        'rasterizor/cuda/rasterize_cuda_kernel.cu',
    ]),
    CUDAExtension('rasterizor.cuda.voxelize', [
        'rasterizor/cuda/voxelize_cuda.cpp',
        'rasterizor/cuda/voxelize_cuda_kernel.cu',
    ]),
]

setup(
    name='rasterizor_pytorch',
    packages=['rasterizor', 'rasterizor.cuda'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)