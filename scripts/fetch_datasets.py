"""
Download the external datasets the agents are trained and gated on.

    python scripts/fetch_datasets.py --source cambray

Each source is downloaded from the authors' own deposit, never from a mirror,
and is reduced on the way in to the columns the project actually uses. The
reduction is what makes the result small enough to work with: Cambray's
integrated measurements are a 306 MB table of 103 columns, of which this
project needs three.

The downloads are large and the extracts are regenerable, so both live under
`data/external/<source>/` and are gitignored. This script is the record of
where they came from, which is the thing worth committing.

Sources

  cambray   Cambray G, Guimaraes JC, Arkin AP, Nat Biotechnol 36:1005 (2018).
            Supplementary Data 15, "Integrated phenotypic measurements", from
            the authors' OSF deposit at doi:10.17605/OSF.IO/A56VU. 244,000
            designed sequences with processed protein levels.

  boel      Boël G et al, Nature 529:358 (2016). Supplementary Data 2, the
            sheet the authors call `E6348.csv`, from the article's own
            supplementary files. One row per gene with the expressed sequence,
            the expression score and the three RNAstructure energies. This is
            the table `data/processed/feature_table_v3.csv` was built from, and
            recovering it is what unblocks identity clustering, the reserved
            critic pool and Gate A.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import urllib.request
import zipfile
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger("SynthBio.fetch_datasets")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.fetch_datasets] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

USER_AGENT = "SynthBio-research/1.0 (dataset ingestion; contact via repository)"

CAMBRAY = {
    "url": "https://osf.io/download/pm2qa/",
    "deposit": "OSF doi:10.17605/OSF.IO/A56VU, Supplementary Data 15",
    "archive": "data/external/cambray2018/raw/Supplementary_Data_15.csv.zip",
    "out": "data/external/cambray2018/variants.csv",
    # Columns in the authors' table. `gs.sequence` is the 96 nt designed region
    # that varies between constructs; everything downstream of it in the
    # reporter is identical across the library. `clean.lin.prot.mean` is the
    # processed protein level, rescaled to a linear measure between 1 and 100
    # and averaged over the replicate FACS-Seq experiments.
    "id_column": "",
    "sequence_column": "gs.sequence",
    "readout_column": "clean.lin.prot.mean",
    "extra_columns": ("seed", "combi", "rep"),
}


def download(url: str, destination: str) -> str:
    """Fetch a file once, skipping the download when it is already on disk."""
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        logger.info(f"Already downloaded: {destination} ({os.path.getsize(destination):,} bytes)")
        return destination

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    logger.info(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=1800) as response, open(destination, "wb") as handle:
        total = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
            if total % (32 << 20) < (1 << 20):
                logger.info(f"  {total / (1 << 20):.0f} MB")
    logger.info(f"Wrote {destination} ({os.path.getsize(destination):,} bytes)")
    return destination


def reduce_cambray(archive: str, out: str) -> int:
    """
    Stream the authors' table out of its zip, keeping three columns.

    Streamed rather than extracted: the table is 306 MB uncompressed and only
    the sequence, the protein level and the design cell are ever used. Rows
    with no protein measurement are dropped here, because a variant with no
    readout cannot take part in a ranking.
    """
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [n for n in bundle.namelist() if n.endswith(".csv") and not n.startswith("__")]
        if not members:
            raise ValueError(f"'{archive}' holds no CSV.")
        member = members[0]
        logger.info(f"Reading {member} from {archive}")

        kept = 0
        with bundle.open(member) as raw, open(out, "w", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
            writer = csv.writer(handle)
            writer.writerow(["variant_id", "sequence", "protein", "seed", "combi", "rep"])
            for index, row in enumerate(reader, start=1):
                readout = (row.get(CAMBRAY["readout_column"]) or "").strip()
                sequence = (row.get(CAMBRAY["sequence_column"]) or "").strip().upper()
                if not sequence or readout in ("", "NA", "NaN"):
                    continue
                writer.writerow([
                    row.get(CAMBRAY["id_column"]) or f"cambray_{index}",
                    sequence,
                    readout,
                    row.get("seed", ""),
                    row.get("combi", ""),
                    row.get("rep", ""),
                ])
                kept += 1
                if index % 50000 == 0:
                    logger.info(f"  {index:,} rows read, {kept:,} kept")
    logger.info(f"Wrote {out}: {kept:,} measured variants.")
    return kept


def fetch_cambray() -> int:
    archive = download(CAMBRAY["url"], CAMBRAY["archive"])
    kept = reduce_cambray(archive, CAMBRAY["out"])
    logger.info(
        "Source: " + CAMBRAY["deposit"] + ". The loader in src/datasets/cambray.py groups these "
        "variants by translated protein, which is the only grouping in this library that gives a "
        "synonymous comparison."
    )
    return kept


BOEL = {
    "url": (
        "https://media.springernature.com/original/springer-static/esm/"
        "art%3A10.1038%2Fnature16509/MediaObjects/41586_2016_BFnature16509_MOESM292_ESM.xlsx"
    ),
    "deposit": "Nature doi:10.1038/nature16509, Supplementary Data 2, sheet E6348.csv",
    "archive": "data/external/boel2016/raw/Supplementary_Data_2.xlsx",
    "out": "data/processed/feature_table_v3.csv",
    "expected_columns": ("name", "E", "Gz48v", "Gave96", "Gave48", "Gave144", "seq"),
}


def fetch_boel() -> int:
    """
    Rebuild `feature_table_v3.csv`, the sequence bearing table, from the paper.

    The project's v3 table is gitignored and was lost from the working tree,
    which took identity clustering, the critic pool and Gate A with it. The
    authors' own supplementary sheet carries the same columns, so the table is
    recoverable rather than gone.

    `scripts/build_feature_table.py` turns the result back into
    `feature_table_v4.csv`, and the sha256 recorded in the artifact's metadata
    is the check that the recovered table is the one the model was fitted on.
    """
    import openpyxl

    archive = download(BOEL["url"], BOEL["archive"])
    workbook = openpyxl.load_workbook(archive, read_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)

    header = [str(c) if c is not None else "" for c in next(rows)]
    missing = [c for c in BOEL["expected_columns"] if c not in header]
    if missing:
        raise ValueError(
            f"'{archive}' sheet '{sheet.title}' is missing {', '.join(missing)}. "
            f"Found {header}. The sheet layout has changed and the mapping must be revisited."
        )

    os.makedirs(os.path.dirname(BOEL["out"]), exist_ok=True)
    written = 0
    with open(BOEL["out"], "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            if row is None or row[0] is None:
                continue
            writer.writerow(["" if value is None else value for value in row])
            written += 1
    logger.info(f"Wrote {BOEL['out']}: {written:,} genes with sequences.")
    logger.info(
        "Source: " + BOEL["deposit"] + ". Rebuild the modelling table with "
        "`python -m scripts.build_feature_table`, and cluster with `python scripts/cluster.py`."
    )
    return written


SOURCES = {"cambray": fetch_cambray, "boel": fetch_boel}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    args = parser.parse_args()
    SOURCES[args.source]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
