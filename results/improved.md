# Improvement: robustness to lighting and rotation

## What we changed

Two changes were made, one for each failure mode. Everything else (descriptor,
distance, split) is unchanged, so the comparison against the baseline is fair.

### 1. Illumination normalisation (lighting)

A one-side light is a smooth multiplicative field `L(x)`: `observed = L * face`.
Blurring the image gives approximately `L * blur(face)`, so dividing the image by its
own Gaussian-blurred copy cancels `L` while keeping the local face texture that LBP
encodes.

- Implemented in `lbpface/preprocess.py` as `normalize_illumination(image, sigma=8)`.
- Applied to every image (gallery and probe) before the LBP step.

### 2. Rotation-augmented gallery (rotation)

At enrolment, rotated copies of every gallery image are stored (`-30` to `+30` degrees
in `5`-degree steps). A rotated probe then matches the correctly oriented template of
its own subject instead of failing against upright templates only.

- Implemented in `experiments/improved.py` as `fix_rotation(images, labels)`.

The evaluation lives in `experiments/improved.py`; the two test-condition helpers
`light(img, s)` and `spin(img, deg)` build the hard probe images.

Reproduce with:

```bash
make improved PY=.venv/bin/python
```

## Results

| Condition     | Baseline | Improved |
|---------------|---------:|---------:|
| normal        |    0.975 |    0.975 |
| uniform       |    0.975 |    0.975 |
| lighting 0.2  |    0.960 |    0.975 |
| lighting 0.4  |    0.945 |    0.970 |
| lighting 0.6  |    0.895 |    0.985 |
| lighting 0.8  |    0.705 |    0.975 |
| lighting 1    |    0.335 |    0.905 |
| rotation 5    |    0.970 |    0.985 |
| rotation 10   |    0.925 |    0.990 |
| rotation 15   |    0.800 |    0.980 |
| rotation 20   |    0.630 |    0.980 |
| rotation 25   |    0.510 |    0.960 |
| rotation 30   |    0.340 |    0.965 |

The accuracy under the change conditions rises clearly, while the normal-photo
accuracy stays at its baseline value (0.975).
