"""The SSRF guard must not reject a public host over a junk IPv6 record.

REPORTED 2026-08-09: "scanning a URL flashes, but no images appear."

The scan itself worked — every THUMBNAIL came back 400, so the grid rendered
with nothing in it. `/api/scrape/thumb` runs `_validate_public_http_url`, and
the image CDN resolved to two public IPv4 addresses **plus** a Teredo IPv6
(`2001::/32`) handed back by the local resolver. Python classifies the whole
Teredo prefix as `is_private`, and `_resolve_public_ips` rejects a host if ANY
resolved address is non-public — so a CDN that answers HTTP 200 over IPv4 was
blocked outright. The same guard sits in `_download_scrape_item`, so importing
those images would have failed too, and returned only a generic 'errors'.

THE FIX, and why it is not a loosening: a Teredo address EMBEDS the client's
real IPv4, so it is judged on that IPv4 — exactly the treatment `_ip_is_blocked`
already gives IPv4-mapped and 6to4 addresses. An attacker's Teredo address
aimed at a LAN host carries that private IPv4 and is still blocked, which is
the first test below. Nothing about the "any bad address rejects the host" rule
changed.
"""
import ipaddress
import socket

import pytest

from app.scrape import netfetch
from app.scrape.netfetch import _ip_is_blocked, _validate_public_http_url


def _teredo_targeting(ipv4: str) -> ipaddress.IPv6Address:
    """A real Teredo address whose embedded client is `ipv4` (obfuscated by the
    XOR-with-all-ones the format specifies)."""
    x = int(ipaddress.IPv4Address(ipv4)) ^ 0xFFFFFFFF
    return ipaddress.ip_address(
        '2001:0:53aa:64c:1c22:c3f8:%x:%x' % (x >> 16, x & 0xFFFF))


# --- the half that must NOT regress ------------------------------------------

@pytest.mark.parametrize('target', ['192.168.1.1', '127.0.0.1', '10.0.0.5',
                                    '169.254.1.1', '172.16.0.1'])
def test_a_teredo_address_aimed_at_an_internal_host_is_still_blocked(target):
    """Unwrapping Teredo must not become a bypass: the embedded IPv4 is the
    machine that would actually be reached."""
    assert _ip_is_blocked(_teredo_targeting(target)) is True


@pytest.mark.parametrize('addr', ['::ffff:10.0.0.1', '2002:c0a8:0101::1',
                                  '::1', 'fe80::1'])
def test_the_other_ipv6_encodings_of_a_private_target_stay_blocked(addr):
    assert _ip_is_blocked(ipaddress.ip_address(addr)) is True


def test_plain_private_and_loopback_ipv4_stay_blocked():
    for a in ('127.0.0.1', '192.168.0.1', '10.1.2.3', '0.0.0.0'):
        assert _ip_is_blocked(ipaddress.ip_address(a)) is True, a


# --- the half that was broken -------------------------------------------------

def test_a_teredo_record_pointing_somewhere_public_is_not_blocked():
    """The measured record: server 0.0.0.0 (a junk/placeholder value), client a
    perfectly ordinary public IPv4."""
    ip = ipaddress.ip_address('2001::4a56:e2ea')
    assert ip.is_private is True, 'precondition: Python calls the prefix private'
    assert ip.teredo[1].is_private is False
    assert _ip_is_blocked(ip) is False


def test_a_public_host_survives_a_junk_teredo_aaaa(monkeypatch):
    """End to end through the guard: two public A records and one Teredo AAAA,
    which is exactly what the reporter's resolver returned for the image CDN."""
    monkeypatch.setattr(netfetch.socket, 'getaddrinfo', lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('45.133.44.50', 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('45.133.44.51', 443)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, '', ('2001::4a56:e2ea', 443, 0, 0)),
    ])
    ok, err = _validate_public_http_url('https://cdn.example.test/a.jpg')
    assert ok is True, err


def test_one_genuinely_internal_address_still_rejects_the_whole_host(monkeypatch):
    """The rule itself is unchanged: a host that resolves to a public address
    AND to loopback is a rebinding setup, not a DNS quirk."""
    monkeypatch.setattr(netfetch.socket, 'getaddrinfo', lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('45.133.44.50', 443)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443)),
    ])
    ok, err = _validate_public_http_url('https://cdn.example.test/a.jpg')
    assert ok is False
    assert 'internal network' in (err or '')
