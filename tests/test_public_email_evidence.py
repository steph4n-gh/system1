"""Guard the public-email experiment's data boundaries, not its quality score."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_email", ROOT / "benchmarks/quality/evaluate_public_email.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def row(identifier, keys, *, collection="original", label="ham"):
    return {"id": identifier, "keys": set(keys), "collection": collection, "label": label, "text": identifier}


def test_mail_features_exclude_label_and_routing_headers():
    text, _ = benchmark.parse_message(
        b"Subject: Meeting tomorrow\nFrom: person@example.com\nX-Spam-Status: Yes\n"
        b"Received: hidden-server.example\nContent-Type: text/plain; charset=utf-8\n\nPlease bring notes.")
    assert text == "Subject: Meeting tomorrow\n\nPlease bring notes."


def test_html_keeps_visible_text_without_script_or_style():
    text, _ = benchmark.parse_message(
        b"Content-Type: text/html\nSubject: Sale\n\n"
        b"<style>secret</style><p>Save &amp; shop</p><script>fetch('tracking')</script>")
    assert text == "Subject: Sale\n\nSave & shop"


def test_plain_mime_part_preferred_and_attachment_not_read():
    raw = (b'MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary="outer"\n\n'
           b'--outer\nContent-Type: multipart/alternative; boundary="inner"\n\n'
           b'--inner\nContent-Type: text/plain\n\nVisible body\n'
           b'--inner\nContent-Type: text/html\n\n<p>HTML alternative</p>\n--inner--\n'
           b'--outer\nContent-Type: text/plain\nContent-Disposition: attachment\n\nPrivate attachment\n--outer--')
    text, _ = benchmark.parse_message(raw)
    assert text == "Subject: \n\nVisible body"


def test_input_is_bounded_and_unknown_charset_decodes():
    text, _ = benchmark.parse_message(b"Content-Type: text/plain; charset=nonexistent\n\n" + b"a" * 9000)
    assert len(text) == 8192


def test_transitive_thread_overlap_never_becomes_evaluation():
    rows = [row("a", ["first"]), row("b", ["first", "second"], collection="later"),
            row("c", ["second"], collection="later"), row("d", ["new"], collection="later")]
    splits, exclusions = benchmark.split_groups(rows)
    assert [r["id"] for r in splits["evaluate"]] == ["d"]
    assert exclusions["later_messages_overlapping_original"] == 2
    assert sum(map(len, splits.values())) == 2
    assert benchmark.split_groups(list(reversed(rows))) == (splits, exclusions)


def test_conflicting_labels_excluded_instead_of_chosen():
    splits, exclusions = benchmark.split_groups([row("a", ["same"]), row("b", ["same"], label="spam")])
    assert not any(splits.values())
    assert exclusions["conflicting_label_messages"] == 2


def test_subject_reply_variants_share_a_group():
    _, left = benchmark.parse_message(b"Subject: Weekly project report 123\n\nOne message")
    _, right = benchmark.parse_message(b"Subject: Re: Weekly project report 456\n\nA different message")
    assert left & right


def test_changed_download_fails_before_parsing(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "mail.tar.bz2").write_bytes(b"changed")
    protocol = {"sources": [{"file": "mail.tar.bz2", "bytes": 7, "sha256": "incorrect"}]}
    with pytest.raises(ValueError, match="checksum mismatch"):
        benchmark.prepare(protocol, tmp_path)


def test_evaluation_requires_frozen_splits(tmp_path):
    with pytest.raises(ValueError, match="Freeze prepared_sha256"):
        benchmark.evaluate({}, {"prepared_sha256": None}, tmp_path)


def test_mixed_collection_split_retains_each_group_once():
    rows = [{"id": str(i), "group": benchmark.digest(str(i))} for i in range(100)]
    mixed = benchmark.representative_split({"teach": rows[:50], "evaluate": rows[50:]})
    assert all(mixed.values())
    assert sorted(r["id"] for items in mixed.values() for r in items) == sorted(r["id"] for r in rows)
    assert mixed == benchmark.representative_split({"evaluate": rows[::-1]})
