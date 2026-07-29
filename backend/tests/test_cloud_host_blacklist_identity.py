"""Blacklisting a bad vast.ai host by machine_id alone does not hold.

Measured on 2026-07-28: run #120 failed on one machine_id and the host was
blacklisted. Run #121 was rented three minutes later on a DIFFERENT machine_id
— at the same public address. A vast machine_id is a file on the host
(/var/lib/vastai_kaalia/machine_id); reinstalling the daemon mints a new one on
the same physical box, so the ban was walked around without anyone trying.

The address is a WEAKER identity than the machine id — several machines can sit
behind one NAT — so it may only widen a ban a real failure already justified,
and it may never be the reason a launch finds nothing. Both properties are
tested here. Offline: no offer search, no rental, no network.
"""
import json

import pytest

# Reused rather than re-created: the launch suite already owns the fixture that
# seeds a trainable dataset.
from tests.test_cloud_training_launch import seeded_dataset  # noqa: F401

# Stand-ins for the two hosts of the incident. The real ids and address are
# deliberately NOT here: a public repo is no place for a third party's machine
# identity, and the shape is all the test needs (the address uses the RFC 5737
# documentation range).
BAD_MACHINE = 10001
NEW_MACHINE = 10002
SHARED_IP = '203.0.113.38'


@pytest.fixture()
def ct(app, monkeypatch):
    from app.services import cloud_training
    monkeypatch.setattr(cloud_training, '_start_monitor', lambda *a, **k: None)
    return cloud_training


def _offer(offer_id, machine_id, ip=None, price=0.47):
    o = {'offer_id': offer_id, 'gpu_name': 'RTX 5090', 'dph_total': price,
         'machine_id': machine_id}
    if ip is not None:
        o['public_ipaddr'] = ip
    return o


def test_a_re_registered_machine_at_the_same_address_is_still_banned(ct, app):
    """THE incident. Same box, new machine_id, same address."""
    with app.app_context():
        ct._blacklist_host(BAD_MACHINE, 'transient pod failure', ip=SHARED_IP)
        kept = ct._filter_offers([
            _offer(1, BAD_MACHINE, SHARED_IP),
            _offer(2, NEW_MACHINE, SHARED_IP),     # run #121's host
            _offer(3, 555, '203.0.113.7'),
        ])
        assert [o['offer_id'] for o in kept] == [3]


def test_an_offer_without_an_address_is_judged_on_its_machine_id_only(ct, app):
    """public_ipaddr is documented on the offer but not guaranteed to be
    populated. An offer that does not carry one must not be dropped — nor
    silently exempted from the machine_id ban."""
    with app.app_context():
        ct._blacklist_host(BAD_MACHINE, 'transient pod failure', ip=SHARED_IP)
        kept = ct._filter_offers([_offer(1, BAD_MACHINE), _offer(2, NEW_MACHINE)])
        assert [o['offer_id'] for o in kept] == [2]


def test_the_address_ban_never_starves_a_launch(ct, app):
    """The wide ban may not be the reason nothing is rentable: when it would
    empty the market, the launch falls back to the narrow machine_id ban."""
    with app.app_context():
        ct._blacklist_host(BAD_MACHINE, 'transient pod failure', ip=SHARED_IP)
        kept = ct._filter_offers([_offer(2, NEW_MACHINE, SHARED_IP)])
        assert [o['offer_id'] for o in kept] == [2]
        # ...but the machine that actually failed stays out, always.
        assert ct._filter_offers([_offer(1, BAD_MACHINE, SHARED_IP)]) == []


def test_a_legacy_blacklist_file_keeps_working(ct, app):
    """Files written before this wave have no 'ip' key. They must load and ban
    exactly what they always banned."""
    with app.app_context():
        ct._bad_hosts_path().parent.mkdir(parents=True, exist_ok=True)
        ct._bad_hosts_path().write_text(
            json.dumps({str(BAD_MACHINE): {'ts': ct._now(), 'reason': 'legacy'}}),
            encoding='utf-8')
        kept = ct._filter_offers([_offer(1, BAD_MACHINE, SHARED_IP),
                                  _offer(2, NEW_MACHINE, SHARED_IP)])
        assert [o['offer_id'] for o in kept] == [2]


def test_the_rented_pods_address_is_what_gets_banned(ct, app):
    """The address recorded from the RENTED instance outranks the one the offer
    advertised — it is the box actually paid for."""
    with app.app_context():
        run = ct.CloudTrainingRun(
            dataset_id=1, status='training', job_name='j1', vast_label='lds-1',
            train_params=json.dumps({'machine_id': BAD_MACHINE,
                                     'offer_ip': '203.0.113.99'}))
        ct.db.session.add(run)
        ct.db.session.commit()
        assert ct._run_host_ip(run) == '203.0.113.99'
        ct._stamp_host_ip(run, SHARED_IP)
        assert ct._run_host_ip(run) == SHARED_IP

        ct._blacklist_run_host(run, 'pod did not become ready in time')
        entry = ct._load_bad_hosts()[str(BAD_MACHINE)]
        assert entry['ip'] == SHARED_IP


def test_provision_stamps_every_identity_the_offer_carried(ct, app, seeded_dataset,
                                                           monkeypatch):
    from tests.test_cloud_training_launch import _fake_export
    monkeypatch.setenv('VAST_API_KEY', 'k-test')
    monkeypatch.setattr(ct, '_reconcile_before_launch', lambda a: None)
    _fake_export(monkeypatch, ct)
    monkeypatch.setattr(ct.vast_client, 'search_offers', lambda **kw: [
        {'offer_id': 9, 'gpu_name': 'RTX 5090', 'dph_total': 0.47,
         'gpu_ram_gb': 32.0, 'machine_id': BAD_MACHINE, 'host_id': 4242,
         'public_ipaddr': SHARED_IP, 'reliability': 0.99}])
    monkeypatch.setattr(ct.vast_client, 'create_instance', lambda *a, **kw: '777')
    with app.app_context():
        ct.launch_cloud_training('local', seeded_dataset)
        run = ct.get_active_run()
        ct._provision(run)
        params = json.loads(run.train_params)
        assert params['machine_id'] == BAD_MACHINE
        assert params['host_id'] == 4242
        assert params['offer_ip'] == SHARED_IP


def test_search_offers_forwards_the_host_identity_fields(monkeypatch):
    """vast_client used to drop host_id and public_ipaddr during the remap, so
    no later layer could ever see them."""
    from app.services import vast_client

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {'offers': [{'id': 9, 'gpu_name': 'RTX 5090', 'dph_total': 0.47,
                                'gpu_ram': 32768, 'machine_id': BAD_MACHINE,
                                'host_id': 4242, 'public_ipaddr': SHARED_IP,
                                'reliability2': 0.99}]}

    monkeypatch.setattr(vast_client, '_request', lambda *a, **kw: _Resp())
    o = vast_client.search_offers(24, 1.0)[0]
    assert (o['machine_id'], o['host_id'], o['public_ipaddr']) == (
        BAD_MACHINE, 4242, SHARED_IP)


def test_an_offer_missing_the_new_fields_degrades_to_none(monkeypatch):
    from app.services import vast_client

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {'offers': [{'id': 9, 'gpu_name': 'RTX 5090', 'dph_total': 0.47,
                                'gpu_ram': 32768, 'machine_id': BAD_MACHINE}]}

    monkeypatch.setattr(vast_client, '_request', lambda *a, **kw: _Resp())
    o = vast_client.search_offers(24, 1.0)[0]
    assert o['host_id'] is None and o['public_ipaddr'] is None
