"""Text-cache status stays readable before optional ML dependencies exist."""
import builtins


def test_cached_query_status_degrades_without_numpy(monkeypatch):
    from app.services import clip_text_encoder

    real_import = builtins.__import__

    def import_without_numpy(name, *args, **kwargs):
        if name == 'numpy':
            raise ModuleNotFoundError("No module named 'numpy'", name='numpy')
        return real_import(name, *args, **kwargs)

    clip_text_encoder.forget_memory_cache()
    monkeypatch.setattr(builtins, '__import__', import_without_numpy)

    assert clip_text_encoder.cached_queries(engine='clip') == 0
    assert clip_text_encoder.cached_queries(engine='siglip2') == 0
