from datetime import date

from traffic_twin.data.download import build_where, export_url


def test_where_clause():
    w = build_where(48.838, 2.361, 600, date(2026, 7, 1))
    assert "geom'POINT(2.361 48.838)'" in w
    assert "600m" in w
    assert "date'2026-07-01'" in w


def test_export_url():
    assert export_url("https://x/api/", "ds") == "https://x/api/ds/exports/parquet"
