"""
generate_dataset_AML.py — EpiModX training data construction pipeline (AML generalization)

Adapts the AD dataset pipeline (generate_dataset.py) to AML vs. healthy myeloid controls
for model generalization testing.

Disease groups:
  - AML     : Acute Myeloid Leukemia patients (Blueprint / IHEC, ERS... accessions)
  - Healthy : Blueprint healthy myeloid controls (neutrophil, CD14+ monocyte, ERS... accessions)

Metadata sources (in AML_datasets/):
  - IHEC_AML_samples-31673.csv          : AML patient sample metadata
  - IHEC_Blueprint_Healthy_samples.csv  : Healthy myeloid control sample metadata

BED file naming convention (must match downloaded/converted files):
  {bigwig_dir}/{HISTONE}_{sample_id}.bed.gz
  e.g. H3K27ac_ERS1023620.bed.gz  (AML)
       H3K27ac_ERS150368.bed.gz   (Healthy)

Pipeline per histone mark:
  1. Load AML + Healthy sample metadata → unified patient list (group_abbrev ∈ {AML, Healthy})
  2. Load per-patient ChIP-seq BED files for the target histone (skip missing)
  3. Merge overlapping peaks (>80% reciprocal overlap) across all patients
  4. Center each merged peak at its midpoint, extend to 4096 bp
  5. Label each region: binary vector (one entry per patient, 1 = has peak)
  6. Filter positives: ≥1 peak in central 2 kb OR >50% of seq length is peak
  7. Sample equal number of negatives (regions devoid of histone peaks),
     excluding ENCODE blacklist regions
  8. Write CSV: chrom, start, end, [patient_col × N]

Output: AML_datasets/{histone}_AML_generated.csv

Usage:
  python generate_dataset_AML.py --histone H3K27ac
  python generate_dataset_AML.py --histone H3K4me3 --blacklist hg38-blacklist.v2.bed.gz
  python generate_dataset_AML.py --histone H3K27me3 --seed 42
"""

import os
import gzip
import argparse
import random
import bisect
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

# Constants

HISTONE_TYPES  = ["H3K27ac", "H3K4me3", "H3K27me3"]
SEQ_LENGTH     = 4096
CENTRAL_WINDOW = 2000
OVERLAP_THRESH = 0.80

# Two clinical groups: AML disease vs. Blueprint healthy myeloid controls
DISEASE_MAP = OrderedDict([
    ("Acute Myeloid Leukemia", "AML"),
    ("Healthy",                "Healthy"),
])

VALID_CHROMS = {f"chr{i}" for i in list(range(1, 23)) + ["X"]}

CHROM_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559,
    "chr4": 190214555, "chr5": 181538259, "chr6": 170805979,
    "chr7": 159345973, "chr8": 145138636, "chr9": 138394717,
    "chr10": 133797422,"chr11": 135086622,"chr12": 133275309,
    "chr13": 114364328,"chr14": 107043718,"chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285,
    "chr19": 58617616, "chr20": 64444167, "chr21": 46709983,
    "chr22": 50818468, "chrX":  156040895,
}


# Metadata loading

def load_metadata_aml(aml_csv: str, cmp_csv: str) -> list:
    """
    Build a unified patient list from the AML and Healthy metadata CSVs.
    Donor deduplication is intentionally deferred to build_dataset_aml, after
    filtering to patients that actually have a BED file for the target histone.
    This ensures we keep the duplicate that has data, not just the first in CSV order.

    Returns a list of dicts with keys:
        group, group_abbrev, sample_id, donor_id, H3K27ac, H3K4me3, H3K27me3
    """
    records = []

    for csv_path, expected_abbrev in [(aml_csv, "AML"), (cmp_csv, "CMP")]:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for _, row in df.iterrows():
            sample_id = str(row["id"]).strip()
            disease   = str(row["disease"]).strip()
            donor_id  = str(row.get("donor_id", sample_id)).strip()
            abbrev    = DISEASE_MAP.get(disease)
            if abbrev is None:
                print(f"[WARN] Unmapped disease label: '{disease}' for {sample_id}")
                continue
            records.append({
                "group":        disease,
                "group_abbrev": abbrev,
                "sample_id":    sample_id,
                "donor_id":     donor_id,
                "H3K27ac":  sample_id,
                "H3K4me3":  sample_id,
                "H3K27me3": sample_id,
            })

    return records


def deduplicate_by_donor(patients: list) -> list:
    """
    Among samples sharing the same (group_abbrev, donor_id), keep only the first.
    Must be called AFTER filtering to samples with actual BED files, so we
    prefer whichever duplicate has data for the target histone.
    """
    seen = set()
    kept, dropped = [], []
    for p in patients:
        key = (p["group_abbrev"], p["donor_id"])
        if key in seen:
            dropped.append(p["sample_id"])
        else:
            seen.add(key)
            kept.append(p)
    if dropped:
        print(f"[DEDUP] Removed {len(dropped)} duplicate-donor samples: {dropped}")
    return kept


# Reused helpers (identical to generate_dataset.py)
# generate_dataset.py is verified by successfully generating identical input as the original study

def load_bed_gz(path: str):
    peaks = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            if chrom in VALID_CHROMS:
                peaks.append((chrom, start, end))
    return peaks


def load_blacklist(path: str):
    if path is None or not os.path.exists(path):
        return {}
    bl = defaultdict(list)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            bl[parts[0]].append((int(parts[1]), int(parts[2])))
    for chrom in bl:
        bl[chrom].sort()
    return dict(bl)


def overlaps_blacklist(chrom: str, start: int, end: int, blacklist: dict) -> bool:
    if chrom not in blacklist:
        return False
    for bl_start, bl_end in blacklist[chrom]:
        if bl_start >= end:
            break
        if bl_end > start:
            return True
    return False


def reciprocal_overlap(a_start, a_end, b_start, b_end) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0:
        return 0.0
    return inter / min(a_end - a_start, b_end - b_start)


def merge_peaks(all_peaks: list, overlap_thresh: float = OVERLAP_THRESH):
    by_chrom = defaultdict(list)
    for chrom, start, end in all_peaks:
        by_chrom[chrom].append((start, end))

    merged = []
    for chrom in sorted(by_chrom):
        intervals = sorted(by_chrom[chrom])
        if not intervals:
            continue
        cur_start, cur_end = intervals[0]
        for start, end in intervals[1:]:
            if reciprocal_overlap(cur_start, cur_end, start, end) >= overlap_thresh:
                cur_start = min(cur_start, start)
                cur_end   = max(cur_end, end)
            else:
                merged.append((chrom, cur_start, cur_end))
                cur_start, cur_end = start, end
        merged.append((chrom, cur_start, cur_end))
    return merged


def center_and_extend(chrom, peak_start, peak_end,
                      seq_length=SEQ_LENGTH, chrom_sizes=CHROM_SIZES):
    mid = (peak_start + peak_end) // 2
    half = seq_length // 2
    new_start = mid - half
    new_end   = new_start + seq_length
    chrom_len = chrom_sizes.get(chrom, 0)
    if new_start < 0 or new_end > chrom_len:
        return None
    return chrom, new_start, new_end


def is_positive(chrom, region_start, region_end,
                patient_peaks_by_chrom: dict,
                seq_length=SEQ_LENGTH, central_window=CENTRAL_WINDOW):
    """
    patient_peaks_by_chrom: dict[pat_idx] → dict[chrom] →
        (starts_list, peaks_list, max_peak_width)

    Forward scan using two bisect bounds:
      lo = first peak that could still overlap region_start (accounts for wide peaks)
      hi = first peak starting at or after region_end (can't overlap)
    This is correct for peaks of any width (including broad H3K27me3 domains).
    """
    center_start = region_start + (seq_length - central_window) // 2
    center_end   = center_start + central_window

    labels = []
    has_central_peak = False
    coverage = np.zeros(seq_length, dtype=np.int8)

    for pat_idx, chrom_peaks in patient_peaks_by_chrom.items():
        pat_label = 0
        if chrom in chrom_peaks:
            starts_list, peaks, max_w = chrom_peaks[chrom]
            # Lower bound: any peak starting before (region_start - max_w) cannot
            # reach region_start even at its maximum possible width.
            lo = bisect.bisect_left(starts_list, region_start - max_w)
            # Upper bound: peaks starting at or after region_end cannot overlap.
            hi = bisect.bisect_left(starts_list, region_end)
            for i in range(lo, hi):
                pk_start, pk_end = peaks[i]
                if pk_end <= region_start:
                    continue  # peak ends before region — skip
                # Peak overlaps [region_start, region_end)
                if pk_start < center_end and pk_end > center_start:
                    has_central_peak = True
                cov_s = max(0, pk_start - region_start)
                cov_e = min(seq_length, pk_end - region_start)
                coverage[cov_s:cov_e] = 1
                pat_label = 1
        labels.append(pat_label)

    frac_covered = int(coverage.sum()) / seq_length
    positive = has_central_peak or (frac_covered > 0.5)
    return positive, labels


def sample_negatives(n_needed: int, all_peak_regions: set,
                     blacklist: dict, seq_length: int = SEQ_LENGTH,
                     rng: random.Random = None):
    if rng is None:
        rng = random.Random(42)

    # Build sorted lists + bisect-ready start arrays per chrom
    occupied_peaks = defaultdict(list)
    for chrom, start, end in all_peak_regions:
        occupied_peaks[chrom].append((start, end))
    for chrom in occupied_peaks:
        occupied_peaks[chrom].sort()
    occupied_starts = {c: [p[0] for p in v] for c, v in occupied_peaks.items()}

    chroms = list(CHROM_SIZES.keys())
    chrom_weights = [CHROM_SIZES[c] for c in chroms]

    negatives = []
    max_attempts = n_needed * 50
    attempts = 0

    while len(negatives) < n_needed and attempts < max_attempts:
        attempts += 1
        chrom = rng.choices(chroms, weights=chrom_weights, k=1)[0]
        chrom_len = CHROM_SIZES[chrom]
        start = rng.randint(0, chrom_len - seq_length)
        end = start + seq_length

        if overlaps_blacklist(chrom, start, end, blacklist):
            continue

        # Bisect: check only intervals near [start, end)
        overlaps = False
        if chrom in occupied_starts:
            peaks = occupied_peaks[chrom]
            starts = occupied_starts[chrom]
            upper = bisect.bisect_left(starts, end)
            for i in range(upper - 1, -1, -1):
                occ_start, occ_end = peaks[i]
                if occ_end <= start:
                    break
                overlaps = True
                break
        if overlaps:
            continue

        negatives.append((chrom, start, end))

    if len(negatives) < n_needed:
        print(f"[WARN] Could only sample {len(negatives)}/{n_needed} negatives "
              f"after {attempts} attempts.")
    return negatives


# Main pipeline

def build_dataset_aml(histone: str, aml_csv: str, cmp_csv: str,
                      bigwig_dir: str, output_dir: str,
                      blacklist_path: str = None,
                      seq_length: int = SEQ_LENGTH, seed: int = 42):
    rng = random.Random(seed)
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Building AML dataset for: {histone}")
    print(f"{'='*60}")

    # 1. Load metadata
    patients = load_metadata_aml(aml_csv, cmp_csv)

    patients_with_data = []
    for p in patients:
        accession = p.get(histone, "")
        bed_path  = os.path.join(bigwig_dir, f"{histone}_{accession}.bed.gz")
        if os.path.exists(bed_path):
            p["bed_path"] = bed_path
            patients_with_data.append(p)
        else:
            print(f"[WARN] BED file not found for {p['sample_id']} / {histone}: {bed_path}")

    print(f"Patients with BED data: {len(patients_with_data)} / {len(patients)}")

    # Deduplicate by donor AFTER BED filter so we keep the sample that has data
    patients_with_data = deduplicate_by_donor(patients_with_data)

    # Sort: AML first, CMP second, then by sample_id
    group_order = list(DISEASE_MAP.values())  # ["AML", "CMP"]
    patients_with_data.sort(
        key=lambda p: (group_order.index(p["group_abbrev"]), p["sample_id"])
    )

    from collections import Counter
    group_counts = Counter(p["group_abbrev"] for p in patients_with_data)
    for g in group_order:
        print(f"  {g}: {group_counts.get(g, 0)} patients")

    if not patients_with_data:
        print("[ERROR] No patients with BED data found. Check bigwig_dir and file names.")
        return None

    # 2. Load all peaks
    print("\nLoading BED files...")
    # patient_peaks[idx][chrom] = (starts_list, peaks_list, max_peak_width)
    # max_peak_width per chrom is used in is_positive to set a safe lower bisect bound,
    # ensuring wide peaks (e.g. H3K27me3 broad domains) are never missed.
    patient_peaks = {}
    all_peaks_flat = []

    for idx, p in enumerate(patients_with_data):
        peaks = load_bed_gz(p["bed_path"])
        by_chrom = defaultdict(list)
        for chrom, start, end in peaks:
            by_chrom[chrom].append((start, end))
        bisect_chrom = {}
        for chrom, intervals in by_chrom.items():
            intervals.sort()
            max_w = max(e - s for s, e in intervals)
            bisect_chrom[chrom] = ([iv[0] for iv in intervals], intervals, max_w)
        patient_peaks[idx] = bisect_chrom
        all_peaks_flat.extend(peaks)
        print(f"  [{idx+1:2d}/{len(patients_with_data)}] {p['sample_id']} "
              f"({p['group_abbrev']}): {len(peaks):,} peaks")

    # 3. Merge overlapping peaks
    print(f"\nMerging {len(all_peaks_flat):,} peaks (overlap threshold={OVERLAP_THRESH})...")
    merged = merge_peaks(all_peaks_flat, OVERLAP_THRESH)
    print(f"  → {len(merged):,} merged peak regions")

    # 4. Center and extend
    print(f"\nCentering and extending to {seq_length} bp...")
    extended = []
    for chrom, peak_start, peak_end in merged:
        region = center_and_extend(chrom, peak_start, peak_end, seq_length)
        if region is not None:
            extended.append(region)
    print(f"  → {len(extended):,} valid regions after boundary check")

    # 5. Load blacklist
    blacklist = load_blacklist(blacklist_path)
    if blacklist:
        print(f"\nBlacklist loaded: {sum(len(v) for v in blacklist.values()):,} regions")
    else:
        print("\nNo blacklist provided — negatives will not be blacklist-filtered")

    # 6. Assign labels; separate positives
    print("\nAssigning labels and identifying positives...")
    positive_rows = []
    positive_regions = set()

    for chrom, start, end in extended:
        if overlaps_blacklist(chrom, start, end, blacklist):
            continue
        is_pos, labels = is_positive(chrom, start, end, patient_peaks, seq_length)
        if is_pos:
            positive_rows.append((chrom, start, end) + tuple(labels))
            positive_regions.add((chrom, start, end))

    print(f"  → {len(positive_rows):,} positive regions")

    # 7. Sample equal negatives
    n_neg = len(positive_rows)
    print(f"\nSampling {n_neg:,} negative regions...")
    neg_regions = sample_negatives(n_neg, positive_regions, blacklist, seq_length, rng)
    print(f"  → {len(neg_regions):,} negative regions sampled")

    n_patients = len(patients_with_data)
    negative_rows = [
        (chrom, start, end) + tuple([0] * n_patients)
        for chrom, start, end in neg_regions
    ]

    # 8. Write output CSV
    patient_cols = [
        f"{p['group_abbrev']}_{p['sample_id']}" for p in patients_with_data
    ]
    columns = ["chrom", "start", "end"] + patient_cols

    all_rows = positive_rows + negative_rows
    df = pd.DataFrame(all_rows, columns=columns)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{histone}_AML_generated.csv")
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"  Total rows    : {len(df):,}  ({len(positive_rows):,} pos / {len(negative_rows):,} neg)")
    print(f"  Columns       : {len(columns)}  (3 coords + {n_patients} patient labels)")
    print(f"    AML cols    : {group_counts.get('AML', 0)}")
    print(f"    Healthy cols: {group_counts.get('Healthy', 0)}")
    print(f"  Chroms     : {sorted(df['chrom'].unique())}")

    return out_path


# Entry point

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate EpiModX AML dataset for one histone mark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--histone", type=str, required=True, choices=HISTONE_TYPES,
        help="Histone mark to process"
    )
    parser.add_argument(
        "--aml_csv", type=str,
        default="./AML_datasets/IHEC_AML_samples-31673.csv",
        help="Path to IHEC AML sample metadata CSV"
    )
    parser.add_argument(
        "--cmp_csv", type=str,
        default="./AML_datasets/IHEC_Blueprint_Healthy_samples.csv",
        help="Path to Blueprint healthy myeloid control sample metadata CSV"
    )
    parser.add_argument(
        "--bigwig_dir", type=str,
        default="./AML_datasets/bigwig",
        help="Directory containing {HISTONE}_{SAMPLE_ID}.bed.gz files "
             "(e.g. H3K27ac_ERS1023620.bed.gz for AML, "
             "H3K27ac_ERS150368.bed.gz for Healthy)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./AML_datasets",
        help="Directory to write {histone}_AML_generated.csv"
    )
    parser.add_argument(
        "--blacklist", type=str, default=None,
        help="Path to ENCODE blacklist BED (optional, e.g. hg38-blacklist.v2.bed.gz)"
    )
    parser.add_argument(
        "--seq_length", type=int, default=SEQ_LENGTH,
        help="Sequence window size in bp"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for negative sampling"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset_aml(
        histone=args.histone,
        aml_csv=args.aml_csv,
        cmp_csv=args.cmp_csv,
        bigwig_dir=args.bigwig_dir,
        output_dir=args.output_dir,
        blacklist_path=args.blacklist,
        seq_length=args.seq_length,
        seed=args.seed,
    )
