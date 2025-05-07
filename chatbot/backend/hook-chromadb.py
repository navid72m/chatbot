from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all data files from chromadb
datas = collect_data_files('chromadb')

# Collect ALL submodules to ensure everything is included
hiddenimports = collect_submodules('chromadb')
hiddenimports = collect_submodules('chromadb.telemetry.product')
hiddenimports = collect_submodules('chromadb.api.segment')
hiddenimports = collect_submodules('chromadb.api.types')
hiddenimports = collect_submodules('chromadb.api.utils')
# hiddenimports = collect_submodules('chromadb.api.types')
# hiddenimports = collect_submodules('chromadb.api.types')
# hiddenimports = collect_submodules('chromadb.api.types')

# Specifically add the embedding functions module
hiddenimports.extend([
    'chromadb.utils.embedding_functions',
    'chromadb.utils.embedding_functions.default',
    'onnxruntime',
    'sentence_transformers',
    'transformers'
])