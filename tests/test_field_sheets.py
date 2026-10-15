"""Printable field sheets: built from the Pitie config and network, one A4 PDF."""

import re

from traffic_twin.validation import field_sheets as fs


def test_page_numbers_follow_the_build_order():
    pg = fs.page_numbers()
    assert pg["cover"] == "1" and pg["map"] == "2"
    assert pg["timing"] == "3-6"  # 10 cycles, 3 per page
    assert pg["counts"] == "7" and pg["transcription"] == "12"


def test_bins_cover_the_simulated_window():
    assert fs.BINS[0] == "07:30" and fs.BINS[-1] == "09:15" and len(fs.BINS) == 8
    assert fs._add15("09:15") == "09:30" and fs._add15("07:45") == "08:00"


def test_build_writes_every_sheet(pitie_scenario, tmp_path):
    out = tmp_path / "sheets.pdf"
    pages = fs.build(pitie_scenario, out, dates="Tue 3 Nov 2026")
    assert pages == 12 and out.stat().st_size > 10_000
    data = out.read_bytes()
    assert len(re.findall(rb"/Type\s*/Page[^s]", data)) == pages
    assert data.startswith(b"%PDF")
