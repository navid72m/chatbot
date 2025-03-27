from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all data files from chromadb
datas = collect_data_files('chromadb')

# Collect ALL submodules to ensure everything is included
hiddenimports = collect_submodules('chromadb')

# Specifically add the embedding functions module
hiddenimports.extend([
    'chromadb.utils.embedding_functions',
    'chromadb.utils.embedding_functions.default',
    'onnxruntime',
    'sentence_transformers',
    'transformers'
])