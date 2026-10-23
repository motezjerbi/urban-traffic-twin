"""English report and slides (D113-D115).

- `facts`: docs/report/facts.json, every quoted result with its source and sha256;
- `assemble`: docs/report/report.md from the templates docs/report/src/NN_*.md;
- `pdf`: docs/report/report.pdf (reportlab, invariant output);
- `acceptance`: the generated A-01..A-10 table;
- `slides`: docs/slides/urban_traffic_twin.pptx (15 slides, python-pptx);
- `figures`: docs/figures/report/*.png; `shots`: docs/figures/app/*.png.

    python -m traffic_twin.report facts | report | slides | figures | shots | check
"""
