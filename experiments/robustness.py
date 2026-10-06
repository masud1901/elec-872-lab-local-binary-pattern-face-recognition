"""Robustness of the LBP face recogniser under lighting and rotation (workshop, Oct 6).

Protocol (taken from the workshop notebook that shipped in ``requirements.txt``):

* data: ORL, 40 subjects x 10 images.
* split: images 1-5 of every subject are the *gallery*, images 6-10 the *probes*
  (deterministic ``np.arange(N) % 10 < 5`` split, no permutations).
* descriptor: ``LBP^{u2}_{16,2}``, 30x37 windows, chi-square, rank-1 rate.

Test conditions
---------------
``normal``             raw probe
``uniform``            ``0.5 * img + 20``  (a global brightness change)
``lighting s``         ``img * (1 + s * linspace(-1, 1, W))`` - one side of the face
                       darker / brighter, ``s`` in ``0.2 .. 1.0`` (the workshop's
                       "one side lit much more strongly" is ``s = 0.8``)
``rotation d``         ``scipy.ndimage.rotate(img, d)`` about the image centre,
                       ``reshape=False``, bilinear

Methods
-------
``baseline``           raw images, plain descriptor
``lighting norm``      flat-field illumination normalisation before LBP
``rotation aug``       gallery augmented with rotated copies of every enrolled image
``improved``           both of the above

The improvement is judged on the *change* conditions: the normal-photo rate must not
drop, while the lit / rotated rates rise.
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from lbpface import repro
from lbpface.classify import recognition_rate
from lbpface.datasets import FaceSet, load_orl
from lbpface.distances import pairwise_distances
from lbpface.features import LBPDescriptor
from lbpface.preprocess import normalize_illumination

# --------------------------------------------------------------------------- #
# test conditions
# --------------------------------------------------------------------------- #
LIGHTING_STRENGTHS = (0.2, 0.4, 0.6, 0.8, 1.0)
ROTATION_ANGLES = (5, 10, 15, 20, 25, 30)
# gallery augmentation angles (must cover the largest tested rotation)
AUG_ANGLES = tuple(range(-max(ROTATION_ANGLES), max(ROTATION_ANGLES) + 1, 5))


def uniform_brightness(images: np.ndarray) -> np.ndarray:
    return np.stack([0.5 * im.astype(np.float64) + 20.0 for im in images])


def lighting_ramp(images: np.ndarray, s: float) -> np.ndarray:
    """Darker on the left, brighter on the right (exactly the workshop transform)."""
    out = []
    for im in images:
        im = im.astype(np.float64)
        ramp = 1.0 + s * np.linspace(-1, 1, im.shape[1])
        out.append(np.clip(im * ramp, 0, 255))
    return np.stack(out)


def rotate(images: np.ndarray, deg: float) -> np.ndarray:
    return np.stack([ndimage.rotate(im.astype(np.float64), deg, reshape=False, order=1,
                                    mode="nearest") for im in images])


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
def _prepare(images: np.ndarray, illum_sigma: Optional[float]) -> np.ndarray:
    if illum_sigma is None:
        return images
    return np.stack([normalize_illumination(im, illum_sigma) for im in images])


def _augment_gallery(images: np.ndarray, labels: np.ndarray, angles: Sequence[float]):
    angles = tuple(angles)
    if not angles:
        return images, labels
    imgs: List[np.ndarray] = []
    labs: List[np.ndarray] = []
    for a in angles:
        rot = rotate(images, a) if a != 0 else images
        imgs.append(rot)
        labs.append(labels)
    return np.concatenate(imgs, axis=0), np.concatenate(labs, axis=0)


class Method:
    """A named recogniser: illumination normalisation + rotation-augmented gallery."""

    def __init__(self, name: str, descriptor: LBPDescriptor,
                 illum_sigma: Optional[float] = None, aug_angles: Sequence[float] = ()):
        self.name = name
        self.descriptor = descriptor
        self.illum_sigma = illum_sigma
        self.aug_angles = tuple(aug_angles)

    def fit(self, gallery: np.ndarray, labels: np.ndarray) -> "Method":
        g = _prepare(gallery, self.illum_sigma)
        g, lab = _augment_gallery(g, labels, self.aug_angles)
        self._G = self.descriptor.transform(g)
        self._labels = lab
        return self

    def score(self, probes: np.ndarray, probe_labels: np.ndarray) -> float:
        p = _prepare(probes, self.illum_sigma)
        P = self.descriptor.transform(p)
        d = pairwise_distances(P, self._G, "chi2")
        return recognition_rate(d, self._labels, probe_labels)

    def describe(self) -> dict:
        d = self.descriptor.describe()
        d.update({"illum_sigma": self.illum_sigma,
                  "aug_angles": list(self.aug_angles)})
        return d


def default_methods(P: int = 16, R: float = 2.0, window=(30, 37)) -> Dict[str, Method]:
    def desc(**kw):
        return LBPDescriptor(P=P, R=R, window=window, **kw)
    return {
        "baseline": Method("baseline", desc()),
        "lighting norm": Method("lighting norm", desc(), illum_sigma=8.0),
        "rotation aug": Method("rotation aug", desc(), aug_angles=AUG_ANGLES),
        "improved": Method("improved", desc(), illum_sigma=8.0, aug_angles=AUG_ANGLES),
    }


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def condition_table(method: Method, probes: np.ndarray, labels: np.ndarray,
                    extra_lighting: bool = True) -> Dict[str, float]:
    """Accuracy of ``method`` on every test condition (rate -> float)."""
    out: Dict[str, float] = {}
    out["normal"] = method.score(probes, labels)
    out["uniform"] = method.score(uniform_brightness(probes), labels)
    for s in LIGHTING_STRENGTHS:
        out[f"lighting {s:g}"] = method.score(lighting_ramp(probes, s), labels)
    for d in ROTATION_ANGLES:
        out[f"rotation {d:g}"] = method.score(rotate(probes, d), labels)
    return out


def split(fs: FaceSet, n_gallery: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic ORL split: the first ``n_gallery`` images of each subject are
    gallery, the rest are probes (the workshop notebook's ``% 10 < 5`` split)."""
    per = fs.images.shape[0] // len(np.unique(fs.subjects))
    idx = np.arange(fs.images.shape[0]) % per
    return idx < n_gallery, idx >= n_gallery


def run(data: str = "data/ORL", seed: int = 0, methods: Optional[Dict[str, Method]] = None,
        verbose: bool = True):
    repro.seed_everything(seed)
    fs = load_orl(data)
    is_gal, is_prb = split(fs)
    gal, prb = fs.images[is_gal], fs.images[is_prb]
    gl, pl = fs.subjects[is_gal], fs.subjects[is_prb]
    methods = methods if methods is not None else default_methods()

    results: Dict[str, dict] = {}
    for name, m in methods.items():
        m.fit(gal, gl)
        results[name] = condition_table(m, prb, pl)
        if verbose:
            print(f"{name:>14s}: " + "  ".join(f"{k} {v:.3f}" for k, v in results[name].items()),
                  flush=True)

    config = {
        "data": "ORL", "seed": seed, "split": "images 1-5 gallery / 6-10 probe",
        "lighting_strengths": list(LIGHTING_STRENGTHS),
        "rotation_angles": list(ROTATION_ANGLES),
        "aug_angles": list(AUG_ANGLES),
        "methods": {n: m.describe() for n, m in methods.items()},
    }
    return config, fs.fingerprint, results


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


BASELINE_C = "#8c8c8c"
IMPROVED_C = "#1f77b4"


def fig_conditions(fs: FaceSet, path: str, subject: int = 1, probe_image: int = 5) -> None:
    """A probe face under the different test conditions (illustration)."""
    plt = _style()
    probe = next(im for im, s, n in zip(fs.images, fs.subjects, fs.names)
                 if s == subject and n.endswith(f"/{probe_image + 5}.pgm"))
    panels = [
        ("normal", probe.astype(np.float64)),
        ("uniform brightness", (0.5 * probe + 20)),
        ("lighting s=0.4", np.clip(probe * (1 + 0.4 * np.linspace(-1, 1, probe.shape[1])), 0, 255)),
        ("lighting s=0.8", np.clip(probe * (1 + 0.8 * np.linspace(-1, 1, probe.shape[1])), 0, 255)),
        ("rotation 10 deg", ndimage.rotate(probe.astype(float), 10, reshape=False, order=1, mode="nearest")),
        ("rotation 20 deg", ndimage.rotate(probe.astype(float), 20, reshape=False, order=1, mode="nearest")),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.1 * len(panels), 3.0))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(f"Workshop test conditions (subject {subject})", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _groups(results: Dict[str, dict]):
    conds = list(next(iter(results.values())).keys())
    lighting = [c for c in conds if c.startswith("lighting")]
    rotation = [c for c in conds if c.startswith("rotation")]
    return conds, lighting, rotation


def fig_summary(results: Dict[str, dict], path: str) -> None:
    """Grouped bar chart: every condition, all methods."""
    plt = _style()
    conds, _, _ = _groups(results)
    methods = list(results)
    x = np.arange(len(conds))
    w = 0.8 / len(methods)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(max(10, 0.72 * len(conds)), 5.2))
    for i, name in enumerate(methods):
        vals = [results[name][c] for c in conds]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=name,
               color=BASELINE_C if name == "baseline" else cmap(i))
    ax.set_xticks(x)
    ax.set_xticklabels(conds, rotation=45, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("rank-1 recognition rate")
    ax.set_title("Accuracy per test condition and method")
    ax.legend(ncol=len(methods), loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_curves(results: Dict[str, dict], path: str) -> None:
    """Two panels: accuracy vs lighting strength, accuracy vs rotation angle."""
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    xs = [0.0] + list(LIGHTING_STRENGTHS)
    for name in ("baseline", "improved"):
        if name not in results:
            continue
        ys = [results[name]["normal"]] + [results[name][f"lighting {s:g}"] for s in LIGHTING_STRENGTHS]
        axes[0].plot(xs, ys, "o-", label=name,
                     color=BASELINE_C if name == "baseline" else IMPROVED_C)
    axes[0].set_xlabel("lighting ramp strength s")
    axes[0].set_ylabel("rank-1 recognition rate")
    axes[0].set_ylim(0.3, 1.02); axes[0].set_title("One-side lighting")
    axes[0].legend()
    xs = [0.0] + list(ROTATION_ANGLES)
    for name in ("baseline", "improved"):
        if name not in results:
            continue
        ys = [results[name]["normal"]] + [results[name][f"rotation {d:g}"] for d in ROTATION_ANGLES]
        axes[1].plot(xs, ys, "o-", label=name,
                     color=BASELINE_C if name == "baseline" else IMPROVED_C)
    axes[1].set_xlabel("rotation (degrees)")
    axes[1].set_ylabel("rank-1 recognition rate")
    axes[1].set_ylim(0.3, 1.02); axes[1].set_title("In-plane rotation")
    axes[1].legend()
    fig.suptitle("Robustness curves: baseline vs improved", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_headline(results: Dict[str, dict], path: str) -> None:
    """Dumbbell chart for the workshop's two headline failure cases + normal."""
    plt = _style()
    cases = [("normal", "normal"), ("lighting 0.8", "one side lit\nmuch more strongly"),
             ("rotation 20", "rotated 20 deg")]
    base = results["baseline"]; imp = results["improved"]
    y = np.arange(len(cases))[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for yi, (key, label) in zip(y, cases):
        b, i = base[key], imp[key]
        ax.plot([b, i], [yi, yi], "-", color="#bbbbbb", lw=3, zorder=1)
        ax.scatter([b], [yi], s=110, color=BASELINE_C, zorder=2, label="baseline" if yi == y[0] else None)
        ax.scatter([i], [yi], s=110, color=IMPROVED_C, zorder=2, label="improved" if yi == y[0] else None)
        ax.annotate(f"{b:.3f}", (b, yi), textcoords="offset points", xytext=(0, 10),
                    ha="center", color="#555555", fontsize=9)
        ax.annotate(f"{i:.3f}", (i, yi), textcoords="offset points", xytext=(0, 10),
                    ha="center", color=IMPROVED_C, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([c[1] for c in cases])
    ax.set_xlim(0.55, 1.03)
    ax.set_xlabel("rank-1 recognition rate")
    ax.set_title("Headline results: baseline vs improved")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_figures(results: Dict[str, dict], out_dir: str, fs: Optional[FaceSet] = None) -> List[str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = [
        os.path.join(out_dir, "fig_summary.png"),
        os.path.join(out_dir, "fig_curves.png"),
        os.path.join(out_dir, "fig_headline.png"),
    ]
    fig_summary(results, paths[0])
    fig_curves(results, paths[1])
    fig_headline(results, paths[2])
    if fs is not None:
        p = os.path.join(out_dir, "fig_conditions.png")
        fig_conditions(fs, p)
        paths.insert(0, p)
    return paths


# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/ORL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/robustness.json")
    ap.add_argument("--figures", default=None, metavar="DIR", help="write PNG figures to DIR")
    a = ap.parse_args(argv)
    config, fp, results = run(a.data, a.seed)
    doc = repro.write_results(a.out, "robustness", config, fp, results)
    print(f"\nwrote {a.out}   digest {doc['result_digest'][:16]}...")
    if a.figures:
        fs = load_orl(a.data)
        for p in make_figures(results, a.figures, fs):
            print(f"figure {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
