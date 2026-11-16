"""Public imports, loaded only when their optional dependencies are needed."""

from importlib import import_module

__all__ = ["trainers", "GFMRetriever", "GraphIndexer"]


def __getattr__(name):
    modules = {
        "trainers": (".trainers", None),
        "GFMRetriever": (".gfmrag_retriever", "GFMRetriever"),
        "GraphIndexer": (".graph_indexer", "GraphIndexer"),
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = modules[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute) if attribute else module
    globals()[name] = value
    return value
