from importlib import import_module

__all__ = ["BaseTextEmbModel", "NVEmbedV2", "Qwen3TextEmbModel"]


def __getattr__(name):
    modules = {
        "BaseTextEmbModel": ".base_model",
        "NVEmbedV2": ".nv_embed",
        "Qwen3TextEmbModel": ".qwen3_model",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
