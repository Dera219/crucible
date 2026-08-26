"""Tests for the extract verifier's format detection.

Only the header sniff is covered here. The verifier's checks themselves run against a real
CRSP extract and cannot be meaningfully faked — but choosing the *loader* is a decision made
before any data is read, and getting it wrong produces an error that blames the extract.
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

CIZ_HEADER = "PERMNO,HdrCUSIP,PrimaryExch,DlyCalDt,DlyPrc,DlyPrcFlg,DlyRet,DlyVol,ShrOut"
SIZ_HEADER = "PERMNO,date,PRC,VOL,RET,SHROUT,SHRCD,EXCHCD,TICKER,DLRET"


def _verifier() -> ModuleType:
    """Import `scripts/verify_extract.py`, which is a script rather than a package member."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "verify_extract.py"
    spec = importlib.util.spec_from_file_location("verify_extract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_extract"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verifier() -> ModuleType:
    return _verifier()


def test_reads_the_header_of_a_plain_csv(verifier: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "daily.csv"
    path.write_text(f"{CIZ_HEADER}\n10001,00462610,A,2015-01-02,13.25,TR,0.01,4200,6110\n")

    assert "dlycaldt" in verifier._header_line(path)


def test_reads_the_header_of_a_gzipped_csv(verifier: ModuleType, tmp_path: Path) -> None:
    """The regression this file exists for.

    WRDS serves compressed output by default and `WRDS.md` recommends gzip for exactly the
    large pulls that most need verifying. Reading a gzip member as text yields replacement
    characters, so `dlycaldt` is not found, the CIZ loader is never tried, and the legacy
    loader refuses the extract for missing `date`/`prc` — columns a CIZ file never has.
    """
    path = tmp_path / "daily.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(f"{CIZ_HEADER}\n10001,00462610,A,2015-01-02,13.25,TR,0.01,4200,6110\n")

    assert "dlycaldt" in verifier._header_line(path)


def test_detects_gzip_by_magic_bytes_not_by_suffix(verifier: ModuleType, tmp_path: Path) -> None:
    """A compressed extract saved without the suffix still has to be read as compressed.

    `_load` deliberately refuses to guess the CRSP format from a filename; sniffing the
    compression from one would reintroduce the same failure a rename away.
    """
    path = tmp_path / "daily_no_suffix.csv"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(f"{CIZ_HEADER}\n10001,00462610,A,2015-01-02,13.25,TR,0.01,4200,6110\n")

    assert "dlycaldt" in verifier._header_line(path)


def test_a_legacy_header_does_not_look_like_ciz(verifier: ModuleType, tmp_path: Path) -> None:
    """The sniff must stay specific, or every SIZ extract routes to the wrong loader."""
    path = tmp_path / "legacy.csv"
    path.write_text(f"{SIZ_HEADER}\n10001,19850102,13.25,4200,0.01,6110,11,1,AMT,\n")

    header = verifier._header_line(path)
    assert "dlycaldt" not in header
    assert "date" in header


def test_an_empty_file_yields_an_empty_header(verifier: ModuleType, tmp_path: Path) -> None:
    """`readline()` on an empty file returns '', which must not raise before the real check."""
    path = tmp_path / "empty.csv"
    path.write_text("")

    assert verifier._header_line(path) == ""
