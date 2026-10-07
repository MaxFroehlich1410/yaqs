# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check scientific data, selection scope and portable artifact generation.

Run ``python -m unittest discover -s tools -t . -p 'test_*.py'`` from the manuscript
directory. Tests use only the packaged data and do not access training archives.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import statistics
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from . import build_figures as figures


class ManuscriptDataTests(unittest.TestCase):
    """Verify endpoint values and comparison semantics against frozen receipts."""

    def setUp(self) -> None:
        """Load the portable data for each independent check."""
        data_path = figures.PACKAGE_ROOT / "data" / "results.json"
        if not data_path.is_file():
            self.skipTest("Manuscript result data is a local artifact and is not distributed with the source.")
        self.data = figures.load_data(data_path)

    def test_supported_coverage_and_archival_counts(self) -> None:
        """Distinguish 120 assessed records from historical depth-search work."""
        endpoints = self.data["logical_endpoints"]
        assert len(endpoints) == 51
        assert len(self.data["native_endpoints"]) == 11
        assert len(self.data["logical_candidates"]) == 120
        assert sum(row["assessed_candidate_count"] for row in endpoints) == 120
        topdown = [row for row in endpoints if row["method"] == figures.TOPDOWN]
        assert len(topdown) == 9
        assert all(row["assessed_candidate_count"] == 1 for row in topdown)
        assert all(row["training_reused"] for row in topdown)
        assert len(self.data["exclusions"]) == 3

    def test_frozen_aggregate_values(self) -> None:
        """Check means and CPU independently of plotting and table formatting."""
        expected = {
            figures.TOPDOWN: (0.9226047691772488, 285.16976000000386),
            "robuststate": (0.910190941165551, 256.515833),
            "fontana": (0.9201037051327098, 31.371391000000006),
            "nacl": (0.40707629047111576, 710.501799),
            "fixed_vqsp": (0.8764935952869579, 115.877722),
            "layerwise": (0.8933174771933413, 354.243756),
        }
        for method, (mean, cpu) in expected.items():
            with self.subTest(method=method):
                rows = [row for row in self.data["logical_endpoints"] if row["method"] == method]
                assert math.isclose(statistics.mean(row["exact_fidelity"] for row in rows), mean, abs_tol=1e-13)
                assert math.isclose(sum(row["build_train_cpu_seconds"] for row in rows), cpu, abs_tol=1e-9)
        assert math.isclose(
            sum(row["build_train_cpu_seconds"] for row in self.data["logical_endpoints"]),
            1753.6802610000036,
            abs_tol=1e-9,
        )

    def test_paired_subsets_are_matched(self) -> None:
        """Restrict Fontana comparisons to the six shared real targets."""
        fontana = figures.paired_gaps(self.data, "fontana")
        robuststate = figures.paired_gaps(self.data, "robuststate")
        assert len(fontana) == 6
        assert not any(target.startswith("haar") for target, _ in fontana)
        assert math.isclose(statistics.mean(gap for _, gap in fontana), 3.8748742561404, abs_tol=1e-12)
        assert len(robuststate) == 9
        assert sum(gap > 0 for _, gap in robuststate) == 5
        assert math.isclose(statistics.mean(gap for _, gap in robuststate), 1.2413828011697756, abs_tol=1e-12)

    def test_pareto_topdown_exceptions(self) -> None:
        """Preserve the two Haar targets where RobustState dominates Top-down."""
        excluded = []
        for target in self.data["target_order"]:
            rows = [row for row in self.data["logical_endpoints"] if row["target"] == target]
            methods = {row["method"] for row in figures.pareto_frontier(rows)}
            if figures.TOPDOWN not in methods:
                excluded.append(target)
        assert excluded == ["haar_random_1", "haar_random_3"]

    @staticmethod
    def test_pareto_requires_strict_improvement() -> None:
        """Keep exact ties and reject only a dominated fidelity/CPU endpoint."""
        rows = [
            {"method": "a", "exact_fidelity": 0.9, "build_train_cpu_seconds": 10.0},
            {"method": "b", "exact_fidelity": 0.9, "build_train_cpu_seconds": 10.0},
            {"method": "c", "exact_fidelity": 0.8, "build_train_cpu_seconds": 10.0},
            {"method": "d", "exact_fidelity": 0.95, "build_train_cpu_seconds": 20.0},
        ]
        assert [row["method"] for row in figures.pareto_frontier(rows)] == ["a", "b", "d"]

    def test_selected_endpoints_are_not_exact_reselected(self) -> None:
        """Retain RobustState's validation-selected critical depth two."""
        selected = next(
            row
            for row in self.data["logical_endpoints"]
            if row["target"] == "tfim_critical" and row["method"] == "robuststate"
        )
        alternatives = [
            row
            for row in self.data["logical_candidates"]
            if row["target"] == "tfim_critical" and row["method"] == "robuststate"
        ]
        assert "depth2" in selected["candidate_id"]
        assert max(row["exact_fidelity"] for row in alternatives) > selected["exact_fidelity"]
        assert math.isclose(
            100 * (max(row["exact_fidelity"] for row in alternatives) - selected["exact_fidelity"]),
            0.05680509165655323,
            abs_tol=1e-12,
        )

    def test_native_rows_are_separate_and_complete(self) -> None:
        """Check native table scores against the native experiment receipts."""
        rows = figures.native_table_rows(self.data)
        assert len(rows) == 11
        assert ["Gaussian", "Top-down", "92.891", "26.17", "14", "10"] in rows
        assert ["Haar 1", "RobustState", "82.090", "52.24", "20", "11"] in rows
        assert not any(row[:2] == ["Haar 1", "Fontana"] for row in rows)
        assert self.data["protocol"]["native_topdown_validation_paths"] == 32768
        assert self.data["protocol"]["validation_paths"] == 8192

    def test_fidelity_table_rounding_and_exclusions(self) -> None:
        """Keep known logical scores and explicit dashes in table output."""
        rows = figures.fidelity_table_rows(self.data)
        assert len(rows) == 9
        assert rows[0] == ["Gaussian", "96.472", "40.346", "94.197", "95.016", "91.246", "94.104"]
        assert sum(row.count(r"\textemdash") for row in rows) == 3

    def test_csv_matches_exact_json_values(self) -> None:
        """Verify the portable spreadsheets contain the same full-precision data."""
        for name in ("logical_endpoints", "native_endpoints"):
            with (figures.PACKAGE_ROOT / "data" / f"{name}.csv").open(newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            json_rows = self.data[name]
            assert len(csv_rows) == len(json_rows)
            for csv_row, json_row in zip(csv_rows, json_rows, strict=True):
                assert csv_row["target"] == json_row["target"]
                assert csv_row["method"] == json_row["method"]
                assert float(csv_row["exact_fidelity"]).hex() == json_row["exact_fidelity"].hex()
                assert float(csv_row["build_train_cpu_seconds"]).hex() == json_row["build_train_cpu_seconds"].hex()

    def test_duplicate_and_invalid_endpoints_are_rejected(self) -> None:
        """Reject accidental duplicate or corrupted records before rendering."""
        cases = [copy.deepcopy(self.data), copy.deepcopy(self.data)]
        cases[0]["logical_endpoints"].append(cases[0]["logical_endpoints"][0])
        cases[1]["native_endpoints"][0]["exact_fidelity"] = 1.01
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for data in cases:
                path.write_text(json.dumps(data))
                error_message = ""
                try:
                    figures.load_data(path)
                except ValueError as error:
                    error_message = str(error)
                assert "Invalid or duplicate endpoint" in error_message

    def test_complete_portable_build_preserves_input(self) -> None:
        """Render every artifact and check formats, dimensions and table content."""
        path = figures.PACKAGE_ROOT / "data" / "results.json"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        figures.configure_style()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            figures.build_workflow(output / "figures")
            figures.build_fidelity(self.data, output / "figures")
            figures.build_pareto(self.data, output / "figures")
            figures.build_tables(self.data, output / "tables")
            for stem in ("workflow", "fidelity", "pareto"):
                assert (output / "figures" / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
                png = (output / "figures" / f"{stem}.png").read_bytes()
                assert png[:8] == b"\x89PNG\r\n\x1a\n"
                width, height = struct.unpack(">II", png[16:24])
                assert width >= 2500
                assert height > 1000
            native = (output / "tables" / "native_results.tex").read_text()
            assert r"\label{tab:native}" in native
            assert "92.891 & 26.17 & 14 & 10" in native
            protocol = (output / "tables" / "protocol.tex").read_text()
            assert "100 online and 100 cross-trajectory" in protocol
            assert "restore best iterate" in protocol
            resources = (output / "tables" / "resources.tex").read_text()
            assert "Gaussian & 7 & 5 & 14 & 10 & 10 & 7 & 10 & 7" in resources
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before


class SourceOnlyCheckoutTests(unittest.TestCase):
    """Verify data-dependent tests handle a source-only checkout."""

    @staticmethod
    def test_missing_local_result_snapshot_skips_data_checks() -> None:
        """Absent excluded result data should skip rather than fail collection."""
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(figures, "PACKAGE_ROOT", Path(directory)),
            pytest.raises(unittest.SkipTest),
        ):
            ManuscriptDataTests().setUp()


if __name__ == "__main__":
    unittest.main()
