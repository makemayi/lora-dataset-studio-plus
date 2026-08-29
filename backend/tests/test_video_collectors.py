"""🧲 video_collectors — the video lane's local-collector socket.

Only what its wave added: the `video_collectors` config section, the runner's
contract (exit code 0 is the whole verdict, `{url}`/`{folder}` substitution,
`@progress` on stderr, stdout ignored), and the two endpoints. The IMAGE
lane's collector shares the spawn/substitute plumbing this feature extracted,
so its JSON contract gets a sanity test here too — a regression in the shared
plumbing would be a regression in code this feature moved.

Every runner test drives a REAL child process (`sys.executable`), because the
contract lives in pipes and exit codes: a fake would verify nothing about
either.
"""
import sys

import pytest

from app import config
from app.services import local_collector


def _configure(command, name='Test collector', lane='video_collectors'):
    config.save_config({lane: {'entries': [{'name': name, 'command': command}]}})


def _touch_script(tmp_path):
    """A child that records its argv INTO `folder` — proving substitution by
    effect rather than by trusting a mock."""
    seen = tmp_path / 'seen'
    script = (
        "import sys, pathlib\n"
        f"pathlib.Path(r'{seen}').write_text('\\n'.join(sys.argv[1:]))\n"
    )
    return seen, [sys.executable, '-c', script, '{url}', '{folder}']


# --- the config section ---------------------------------------------------------

def test_defaults_carry_an_empty_video_collectors_section():
    assert config.get('video_collectors.entries') == []


def test_half_typed_entries_are_dropped_not_reported():
    # A hand-written settings file must not let one broken row take the good
    # ones down — the same stance the image lane takes.
    config.save_config({'video_collectors': {'entries': [
        {'command': ['node', 'x.mjs']},                    # no name
        {'name': 'no command'},
        {'name': 'ok', 'command': ['node', 'x.mjs', '{url}']},
        'not even a dict',
    ]}})
    assert [c['name'] for c in local_collector.configured_video_collectors()] == ['ok']


# --- the runner's contract ------------------------------------------------------

def test_url_and_folder_tokens_are_substituted(tmp_path):
    seen, command = _touch_script(tmp_path)
    folder = tmp_path / 'src'
    folder.mkdir()
    _configure(command)
    local_collector.run_video_collector('Test collector', 'https://site.example/a',
                                        folder=str(folder))
    url_line, folder_line = seen.read_text().split('\n')
    assert url_line == 'https://site.example/a'
    assert folder_line == str(folder)


def test_a_missing_url_token_still_receives_the_url_last(tmp_path):
    seen, (command) = _touch_script(tmp_path)
    command = [sys.executable, '-c',
               command[2], '{folder}']            # same script, no {url} token
    folder = tmp_path / 'src'
    folder.mkdir()
    _configure(command)
    local_collector.run_video_collector('Test collector', 'https://site.example/b',
                                        folder=str(folder))
    assert seen.read_text().split('\n')[1] == 'https://site.example/b'


def test_nonzero_exit_is_an_error_carrying_stderr(tmp_path):
    _configure([sys.executable, '-c',
                "import sys; sys.stderr.write('boom'); sys.exit(3)"])
    with pytest.raises(local_collector.CollectorError) as e:
        local_collector.run_video_collector('Test collector', 'https://site.example/c')
    assert 'exited with code 3' in str(e.value)
    assert 'boom' in e.value.detail


def test_progress_lines_reach_the_callback(tmp_path):
    _configure([sys.executable, '-c',
                "import sys, time\n"
                "sys.stderr.write('@progress {\"done\": 1, \"total\": 2, "
                "\"detail\": \"walking\"}\\n'); sys.stderr.flush()\n"
                "time.sleep(0.2)\n"])
    seen = []
    local_collector.run_video_collector('Test collector', 'https://site.example/d',
                                        on_progress=lambda *a: seen.append(a))
    assert (1, 2, 'walking') in seen


def test_stdout_is_ignored_not_parsed(tmp_path):
    # The video contract's whole point: noise on stdout cannot fail a run,
    # because nothing reads it.
    _configure([sys.executable, '-c',
                "import sys\nprint('this is not JSON {')\n"])
    local_collector.run_video_collector('Test collector', 'https://site.example/e')


def test_a_non_http_url_is_refused_before_any_process_starts():
    _configure([sys.executable, '-c', 'pass'])
    with pytest.raises(local_collector.CollectorError, match='http'):
        local_collector.run_video_collector('Test collector', 'not a url')


def test_an_unconfigured_name_is_refused():
    with pytest.raises(local_collector.CollectorError, match='no collector named'):
        local_collector.run_video_collector('Never configured', 'https://site.example/f')


# --- the shared plumbing: the image lane's contract still holds -----------------

def test_image_lane_still_parses_its_json_contract():
    _configure([sys.executable, '-c',
                "import json, sys\n"
                "print(json.dumps({'items': [{'url': 'https://cdn.example/1.jpg',"
                " 'title': 'One'}], 'suggested_name': 'Account'}))"],
               name='Image collector', lane='collectors')
    items, suggested = local_collector.run_collector(
        'Image collector', 'https://site.example/g')
    assert [i['title'] for i in items] == ['One']
    assert suggested == 'Account'


# --- the endpoints --------------------------------------------------------------

@pytest.fixture()
def bank_folder(tmp_path):
    # Deliberately NOT created: the create endpoint makes it, which is the
    # collector workflow's own first step (an empty bank, then a download).
    return tmp_path / 'bank-src'


@pytest.fixture()
def bank_id(client, bank_folder):
    res = client.post('/api/video-bank/create',
                      json={'name': 'Rushes', 'folder': str(bank_folder)})
    assert res.status_code == 200
    return res.get_json()['id']


def test_the_list_endpoint_starts_empty_and_reflects_config(client):
    assert client.get('/api/video-bank/collectors').get_json() == {'collectors': []}
    _configure([sys.executable, '-c', 'pass'], name='Kuaishou videos')
    assert client.get('/api/video-bank/collectors').get_json() == {
        'collectors': ['Kuaishou videos']}


def test_collect_refuses_an_unknown_bank(client):
    _configure([sys.executable, '-c', 'pass'])
    res = client.post('/api/video-bank/999999/collect',
                      json={'collector': 'Test collector', 'url': 'https://x.example/1'})
    assert res.status_code == 404


def test_collect_refuses_an_unconfigured_collector(client, bank_id):
    res = client.post(f'/api/video-bank/{bank_id}/collect',
                      json={'collector': 'Not configured', 'url': 'https://x.example/2'})
    assert res.status_code == 400
    assert 'no collector named' in res.get_json()['error']


def test_collect_runs_the_collector_and_the_folder_lands_in_the_bank(
        client, bank_id, bank_folder, tmp_path):
    # The end-to-end shape: the command "downloads" a clip into `{folder}`,
    # the job refreshes the bank, and the file shows up as a source — all
    # synchronously, because under TESTING bank jobs run inline.
    seen, command = _touch_script(tmp_path)
    _configure([sys.executable, '-c', command[2], '{folder}', 'write-clip'])
    # The script above writes argv to `seen`; a real collector writes a FILE.
    # Compose instead: one child that does both, so the assertion covers the
    # substitution AND the refresh seeing the result.
    clip = bank_folder / 'clip.mp4'
    config.save_config({'video_collectors': {'entries': [{
        'name': 'Test collector',
        'command': [sys.executable, '-c',
                    "import sys, pathlib\n"
                    f"pathlib.Path(r'{seen}').write_text(sys.argv[2])\n"
                    f"pathlib.Path(r'{clip}').write_bytes(b'x')\n",
                    '{url}', '{folder}'],
    }]}})
    res = client.post(f'/api/video-bank/{bank_id}/collect',
                      json={'collector': 'Test collector', 'url': 'https://site.example/h'})
    assert res.status_code == 202
    assert res.get_json()['ok'] is True
    assert seen.read_text() == str(bank_folder)      # {folder} was THIS bank's
    payload = client.get(f'/api/video-bank/{bank_id}?refresh=1').get_json()
    names = [s['relpath'] if 'relpath' in s else s.get('filename', '')
             for s in payload.get('sources', [])]
    assert any('clip.mp4' in n for n in names)
