from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, copy_metadata

# PyTorch needs its package metadata
datas = copy_metadata('torch')

# Collect all package contents
data_torch, binaries_torch, hiddenimports_torch = collect_all('torch')
datas.extend(data_torch)

# Initialize binaries list before extending it
binaries = []
binaries.extend(binaries_torch)

# Add C extensions explicitly
hiddenimports = hiddenimports_torch + [
    'torch._C',
    'torch.ops',
    'torch.jit',
    'torch.cuda',
    'torch.autograd',
    'transformers',
    'sentence_transformers',
    'chromadb'
]