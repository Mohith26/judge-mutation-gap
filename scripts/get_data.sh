#!/usr/bin/env bash
# Fetch MEDEC-MS (CC BY 4.0, Ben Abacha et al., ACL Findings 2025) at a pinned commit.
set -euo pipefail
SHA=e9173f861b39706946b6a127394992c00535b1e3
mkdir -p data
for f in MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv; do
  curl -sSfL "https://raw.githubusercontent.com/abachaa/MEDEC/$SHA/MEDEC-MS/$f" -o "data/$f"
done
shasum -a 256 -c <<'SUMS'
f35e2c3c1e4d98ca900394c4fcff14c2a84752e48ddc83698401da6a360c707e  data/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv
7571aec311e2fb88d9084082af2323aa579d102e1d5e1a06faf930fe4fd3977d  data/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv
SUMS
