"""Provenance des images de recherche web : la page d'origine est conservée.

Pexels exige un crédit photographe ; une image trouvée sur le web ouvert n'en a
pas — seulement la page où elle a été trouvée. Les deux formes coexistent."""
from app.services.face_dataset_service import (
    _source_metadata_from_scrape_item, normalize_source_metadata)


def test_websearch_metadata_keeps_the_page_the_image_came_from():
    assert normalize_source_metadata({
        'platform': 'websearch',
        'source_url': 'https://blog.example.test/post/42',
    }) == {'platform': 'websearch', 'source_url': 'https://blog.example.test/post/42'}


def test_websearch_metadata_without_a_usable_source_url_is_dropped():
    for value in ('http://blog.example.test/post/42',      # pas https
                  'https://user:pw@blog.example.test/x',   # credentials
                  '', None, 12):
        assert normalize_source_metadata(
            {'platform': 'websearch', 'source_url': value}) is None


def test_websearch_metadata_with_control_characters_is_dropped():
    for value in ('https://blog.example.test/post\x0042',
                  'https://blog.example.test/post/42\nmore',
                  'https://blog.example.test/\x01post/42'):
        assert normalize_source_metadata(
            {'platform': 'websearch', 'source_url': value}) is None


def test_a_websearch_scan_item_yields_its_provenance():
    item = {'url': 'https://cdn.example.test/photo.jpg', 'platform': 'websearch',
            'source_url': 'https://blog.example.test/post/42', 'title': 'x'}
    assert _source_metadata_from_scrape_item(item) == {
        'platform': 'websearch', 'source_url': 'https://blog.example.test/post/42'}


def test_unknown_platforms_are_still_dropped():
    assert normalize_source_metadata(
        {'platform': 'pornpics', 'source_url': 'https://example.test/x'}) is None
