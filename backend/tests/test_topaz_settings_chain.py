def test_settings_route_persists_upscale_engine(client, app):
    """The Settings page's save (PUT /api/settings, section 'engines') must
    persist engines.upscale_engine so the dispatcher sees it."""
    import app.config as cfg

    # reset
    saved = cfg.load_config()
    saved.get('engines', {}).pop('upscale_engine', None)
    cfg.save_config(saved)

    assert cfg.get('engines.upscale_engine') is None

    r = client.put('/api/settings', json={
        'config': {'engines': {'upscale_engine': 'topaz'}}})
    assert r.status_code == 200, r.get_json()
    assert cfg.get('engines.upscale_engine') == 'topaz'

    # and the GET envelope carries it back to the settings page
    d = client.get('/api/settings').get_json()
    assert d['config']['engines'].get('upscale_engine') == 'topaz'
