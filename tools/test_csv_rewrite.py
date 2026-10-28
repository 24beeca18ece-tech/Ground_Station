"""
tools/test_csv_rewrite.py
=========================

The newest-first CSV rewrite copies the journal's lines backwards in blocks
instead of parsing every row into memory. It must produce exactly the file
the old parse-and-reserialise rewrite produced, and its memory must not grow
with the journal.

    python -m pytest tools/test_csv_rewrite.py -v
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tracemalloc

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv_logger  # noqa: E402
from csv_logger import CsvLoggerThread  # noqa: E402
from telemetry_packet import CSV_HEADER  # noqa: E402


def header_line():
    buf = io.StringIO()
    csv.writer(buf).writerow(CSV_HEADER)
    return buf.getvalue()


def row(i):
    """A row shaped like the real ones: numbers, names, and a raw frame that
    needs quoting (commas, and a doubled quote)."""
    values = [i, "2026-10-28T12:00:%02d.000+00:00" % (i % 60), 1.79e9 + i,
              "RX1" if i % 2 else "RX2", 1, "2025ASI001", "CANSAT"]
    values += [round(i * 0.05, 2)] * (len(CSV_HEADER) - len(values) - 1)
    values.append('$2025ASI001,CANSAT,%d,1.0,"q",2.0*4D' % i)
    return values


def write_journal(path, rows, header=True, tail_bytes=b""):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    if tail_bytes:
        with open(path, "ab") as handle:
            handle.write(tail_bytes)


def old_rewrite(journal_path):
    """The pre-change rewrite, verbatim in effect: parse, reverse, write."""
    with open(journal_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        rows = [first] if first is not None and first != CSV_HEADER else []
        rows.extend(r for r in reader if r)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(CSV_HEADER)
    writer.writerows(reversed(rows))
    return out.getvalue().encode("utf-8")


def new_rewrite(tmp_path, journal_path):
    logger = CsvLoggerThread(log_dir=str(tmp_path))
    logger._journal_path = journal_path
    logger._journal_file = None
    logger._csv_path = str(tmp_path / "Flight_test.csv")
    assert logger._rewrite_csv()
    with open(logger._csv_path, "rb") as handle:
        return handle.read()


CASES = {
    "rows": dict(rows=[row(i) for i in range(1, 400)]),
    "header only": dict(rows=[]),
    "no header (first line is data)": dict(rows=[row(i) for i in range(5)],
                                           header=False),
    "unterminated last row": dict(rows=[row(i) for i in range(30)],
                                  tail_bytes=b"999,partial,row"),
    "cut between CR and LF": dict(rows=[row(i) for i in range(30)],
                                  tail_bytes=b"999,partial,row\r"),
    "blank lines": dict(rows=[row(1), [], row(2), [], row(3)]),
}


@pytest.mark.parametrize("block", [csv_logger.REWRITE_BLOCK_BYTES, 64, 7])
@pytest.mark.parametrize("name", list(CASES))
def test_same_bytes_as_the_old_rewrite(tmp_path, monkeypatch, name, block):
    monkeypatch.setattr(csv_logger, "REWRITE_BLOCK_BYTES", block)
    journal = str(tmp_path / "Flight_test.csv.journal")
    write_journal(journal, **CASES[name])
    assert new_rewrite(tmp_path, journal) == old_rewrite(journal)


def test_a_record_spanning_lines_falls_back_to_parsing(tmp_path):
    journal = str(tmp_path / "Flight_test.csv.journal")
    rows = [row(1), row(2)[:-1] + ["two\nlines"], row(3)]
    write_journal(journal, rows)
    logger = CsvLoggerThread(log_dir=str(tmp_path))
    logger._journal_path = journal
    logger._journal_file = None
    assert logger._write_reversed_journal(str(tmp_path / "x.tmp")) is False
    assert new_rewrite(tmp_path, journal) == old_rewrite(journal)


def test_rewrite_memory_does_not_grow_with_the_journal(tmp_path):
    """48,000 rows is 40 minutes at 20 Hz. The old rewrite held every parsed
    row at once (a 42.8 MB peak measured at this size); the line copy holds
    one block."""
    journal = str(tmp_path / "Flight_test.csv.journal")
    write_journal(journal, (row(i) for i in range(48000)))
    logger = CsvLoggerThread(log_dir=str(tmp_path))
    logger._journal_path = journal
    logger._journal_file = None
    logger._csv_path = str(tmp_path / "Flight_test.csv")
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        assert logger._rewrite_csv()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 3 * csv_logger.REWRITE_BLOCK_BYTES + 512 * 1024, peak
    with open(logger._csv_path, "rb") as handle:
        assert handle.read() == old_rewrite(journal)
