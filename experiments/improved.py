"""Workshop improvement: keep LBP face recognition accurate when the lighting
changes across the face and when the face is rotated.

Only two changes are made, each a small function below:

  change 1  `fix_lighting`   - normalise the illumination before LBP
  change 2  `fix_rotation`   - store rotated copies of the gallery images

Run:  python -m experiments.improved --data data/ORL
"""
import argparse

import numpy as np
from scipy import ndimage

from lbpface.classify import recognition_rate
from lbpface.datasets import load_orl
from lbpface.distances import pairwise_distances
from lbpface.features import LBPDescriptor
from lbpface.preprocess import normalize_illumination

DESC = LBPDescriptor.orl()          # LBP(16,2), 30x37 windows, chi2
ANGLES = range(-30, 31, 5)          # gallery rotations to store
ILLUM_SIGMA = 8.0                   # blur scale of the lighting normalisation
LIGHTINGS = (0.2, 0.4, 0.6, 0.8, 1.0)   # one-side lighting strengths to test
ROTATIONS = (5, 10, 15, 20, 25, 30)     # rotation angles to test


# --- the test conditions (from the workshop) --------------------------------
def brighten(img):
    """Uniform brightness change: 0.5 * img + 20."""
    return 0.5 * img + 20


def light(img, s):
    """One side of the face darker / brighter by a linear ramp of strength s."""
    return np.clip(img * (1 + s * np.linspace(-1, 1, img.shape[1])), 0, 255)


def spin(img, deg):
    """Rotate the face by `deg` degrees about the image centre."""
    return ndimage.rotate(img.astype(float), deg, reshape=False, order=1, mode="nearest")


# --- the two changes to the method -----------------------------------------
def fix_lighting(images):
    """Change 1: divide out the smooth lighting field before LBP."""
    return [normalize_illumination(im, ILLUM_SIGMA) for im in images]


def fix_rotation(images, labels):
    """Change 2: add a rotated copy of every gallery image for each angle."""
    rot = [spin(im, a) for a in ANGLES for im in images]
    lab = [l for _ in ANGLES for l in labels]
    return images + rot, labels + lab


# --- recognition ------------------------------------------------------------
def accuracy(gallery, gallery_labels, probes, probe_labels):
    """Rank-1 rate of the probes against the gallery."""
    G = DESC.transform(gallery)
    P = DESC.transform(probes)
    return recognition_rate(pairwise_distances(P, G, "chi2"), gallery_labels, probe_labels)


def conditions(probes):
    """Every test condition applied to the raw probe images."""
    rows = {"normal": probes, "uniform": [brighten(im) for im in probes]}
    for s in LIGHTINGS:
        rows[f"lighting {s:g}"] = [light(im, s) for im in probes]
    for d in ROTATIONS:
        rows[f"rotation {d:g}"] = [spin(im, d) for im in probes]
    return rows


def evaluate(gallery, gallery_labels, probes, probe_labels, use_fixes):
    """Accuracy on every condition, with or without the two fixes."""
    if use_fixes:
        gallery = fix_lighting(gallery)
        gallery, gallery_labels = fix_rotation(gallery, gallery_labels)
    results = {}
    for name, ims in conditions(probes).items():
        if use_fixes:
            ims = fix_lighting(ims)          # fix is applied to the condition image
        results[name] = accuracy(gallery, gallery_labels, ims, probe_labels)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/ORL")
    ap.add_argument("--baseline", action="store_true",
                    help="print only the baseline column")
    a = ap.parse_args(argv)

    fs = load_orl(a.data)
    is_gal = (np.arange(len(fs.images)) % 10) < 5          # images 1-5 = gallery
    gallery, gallery_labels = list(fs.images[is_gal]), list(fs.subjects[is_gal])
    probes, probe_labels = list(fs.images[~is_gal]), list(fs.subjects[~is_gal])

    base = evaluate(gallery, gallery_labels, probes, probe_labels, use_fixes=False)
    if a.baseline:
        print(f"{'condition':<14}{'baseline':>10}")
        print("-" * 24)
        for c in base:
            print(f"{c:<14}{base[c]:>10.3f}")
        return 0

    imp = evaluate(gallery, gallery_labels, probes, probe_labels, use_fixes=True)
    print(f"{'condition':<14}{'baseline':>10}{'improved':>10}")
    print("-" * 34)
    for c in base:
        print(f"{c:<14}{base[c]:>10.3f}{imp[c]:>10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
