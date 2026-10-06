# Baseline results

Method: `LBP^{u2}_{16,2}`, 30x37 windows, chi-square distance, rank-1
nearest-neighbour identification.

Data: ORL (40 subjects x 10 images). Split: for every subject, images 1-5 are the
gallery and images 6-10 the probes.

Reproduce with:

```bash
make baseline PY=.venv/bin/python
```

| Condition        | Rank-1 accuracy |
|------------------|----------------:|
| normal           |           0.975 |
| uniform          |           0.975 |
| lighting 0.2     |           0.960 |
| lighting 0.4     |           0.945 |
| lighting 0.6     |           0.895 |
| lighting 0.8     |       **0.705** |
| lighting 1       |           0.335 |
| rotation 5       |           0.970 |
| rotation 10      |           0.925 |
| rotation 15      |           0.800 |
| rotation 20      |       **0.630** |
| rotation 25      |           0.510 |
| rotation 30      |           0.340 |

The baseline is accurate on normal photos but breaks down under strong one-side
lighting (`lighting 0.8` = 0.705) and rotation (`rotation 20` = 0.630).
