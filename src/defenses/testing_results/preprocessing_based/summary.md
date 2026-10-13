# Preprocessing defences, alone and chained (fixed settings)

Settings: jpeg {'quality': 75}, median {'kernel': 3}, gaussian {'sigma': 8, 'samples': 32, 'seed': 4266}.
Sorted by the worst of the three accuracies. Chains are applied left to right.
Clean = test/ (unattacked), PGD = pgd_attacked/, Pixle = pixle_attacked/; 115 images each.

| Rank | Chain | Defences | Clean accuracy | PGD accuracy | Pixle accuracy | Clean macro F1 | PGD macro F1 | Pixle macro F1 | Worst accuracy |
|---|---|---|---|---|---|---|---|---|---|
| 1 | jpeg → median → gaussian | 3 | 0.9826 | 0.8696 | 0.9826 | 0.9818 | 0.8881 | 0.9818 | 0.8696 |
| 2 | median → jpeg → gaussian | 3 | 0.9826 | 0.8261 | 0.9913 | 0.9818 | 0.8523 | 0.9914 | 0.8261 |
| 3 | jpeg → gaussian → median | 3 | 1.0000 | 0.8000 | 1.0000 | 1.0000 | 0.8289 | 1.0000 | 0.8000 |
| 4 | gaussian → median → jpeg | 3 | 0.9913 | 0.7652 | 1.0000 | 0.9914 | 0.7977 | 1.0000 | 0.7652 |
| 5 | jpeg → gaussian | 2 | 0.9826 | 0.7478 | 0.9826 | 0.9818 | 0.7753 | 0.9818 | 0.7478 |
| 6 | median → gaussian → jpeg | 3 | 0.9913 | 0.7304 | 0.9913 | 0.9914 | 0.7702 | 0.9914 | 0.7304 |
| 7 | gaussian → jpeg → median | 3 | 1.0000 | 0.7304 | 1.0000 | 1.0000 | 0.7675 | 1.0000 | 0.7304 |
| 8 | median → gaussian | 2 | 0.9913 | 0.5304 | 0.9913 | 0.9914 | 0.5447 | 0.9914 | 0.5304 |
| 9 | gaussian → jpeg | 2 | 0.9826 | 0.4957 | 0.9739 | 0.9818 | 0.5261 | 0.9757 | 0.4957 |
| 10 | gaussian → median | 2 | 1.0000 | 0.4957 | 1.0000 | 1.0000 | 0.5087 | 1.0000 | 0.4957 |
| 11 | jpeg → median | 2 | 1.0000 | 0.4870 | 1.0000 | 1.0000 | 0.4968 | 1.0000 | 0.4870 |
| 12 | median → jpeg | 2 | 0.9913 | 0.4609 | 0.9826 | 0.9914 | 0.4762 | 0.9847 | 0.4609 |
| 13 | jpeg | 1 | 0.9913 | 0.1739 | 0.9565 | 0.9914 | 0.1938 | 0.9452 | 0.1739 |
| 14 | gaussian | 1 | 0.9826 | 0.0348 | 0.9652 | 0.9818 | 0.0563 | 0.9685 | 0.0348 |
| 15 | median | 1 | 1.0000 | 0.0261 | 1.0000 | 1.0000 | 0.0357 | 1.0000 | 0.0261 |

## Analysis

Written by hand on 2026-10-14 from the table above and `summary.csv` (fixed settings: JPEG quality 75, median 3 x 3, Gaussian smoothing σ = 8 with 32 samples). The script keeps this section when it regenerates the table, so update it if the results change.

Each test set has 115 images, so one image is about 0.87 points.

### All 15 pipelines

The no-defence row is Phase 2's result for the undefended classifier (`src/modelling/test.py`), shown for reference. This script does not test it.

| Pipeline | Clean | PGD | Pixle | Wrong (clean / PGD / Pixle) |
|---|---|---|---|---|
| no defence (Phase 2 result) | 100% | 0.0% | 60.0% | 0 / 115 / 46 |
| jpeg | 99.1% | 17.4% | 95.7% | 1 / 95 / 5 |
| gaussian | 98.3% | 3.5% | 96.5% | 2 / 111 / 4 |
| median | 100% | 2.6% | 100% | 0 / 112 / 0 |
| jpeg → gaussian | 98.3% | 74.8% | 98.3% | 2 / 29 / 2 |
| median → gaussian | 99.1% | 53.0% | 99.1% | 1 / 54 / 1 |
| gaussian → jpeg | 98.3% | 49.6% | 97.4% | 2 / 58 / 3 |
| gaussian → median | 100% | 49.6% | 100% | 0 / 58 / 0 |
| jpeg → median | 100% | 48.7% | 100% | 0 / 59 / 0 |
| median → jpeg | 99.1% | 46.1% | 98.3% | 1 / 62 / 2 |
| jpeg → median → gaussian | 98.3% | **87.0%** | 98.3% | 2 / 15 / 2 |
| median → jpeg → gaussian | 98.3% | 82.6% | 99.1% | 2 / 20 / 1 |
| jpeg → gaussian → median | **100%** | 80.0% | **100%** | 0 / 23 / 0 |
| gaussian → median → jpeg | 99.1% | 76.5% | 100% | 1 / 27 / 0 |
| median → gaussian → jpeg | 99.1% | 73.0% | 99.1% | 1 / 31 / 1 |
| gaussian → jpeg → median | 100% | 73.0% | 100% | 0 / 31 / 0 |

### Clean accuracy: what the defences cost

- **Small cost.** Every pipeline scores 98.3% to 100% on clean images. The average is 99.1% whether one, two or three defences are used, so stacking adds no extra cost.
- **Only two images ever fail.** Across all 15 defended pipelines, the clean errors are the same two images:
  - `30s8_png.rf.00f20f38ebe377cbcb0712b7061faaa7.jpg`, a 30 mph sign read as 50 (in 10 pipelines)
  - `60_27.jpg`, a 60 mph sign read as 70 (in 5 pipelines)
- **The last step decides it.**
  - All five pipelines that end with the median filter are 100% clean.
  - Pipelines ending in Gaussian smoothing lose 1 or 2 images.
  - Pipelines ending in JPEG lose 1 or 2 images.

### PGD accuracy: stacking and order

| Defences in the chain | Average PGD accuracy | Range |
|---|---|---|
| 0 | 0.0% | n/a |
| 1 | 7.8% | 2.6% to 17.4% |
| 2 | 53.6% | 46.1% to 74.8% |
| 3 | 78.7% | 73.0% to 87.0% |

- **Depth matters most.** No single defence gets past 17%. The worst pair (46.1%) still beats the best single (17.4%), but the best pair (74.8%) beats two of the six triples.
- **Order matters next.**
  - JPEG works best first: 83% average as the first step of a triple, versus 75% as the last.
  - Gaussian smoothing works best last: 85% average as the last step, versus 75% as the first.
  - The clearest case is the pair jpeg → gaussian at 74.8%, against the reverse order at 49.6%.
- **The gain is spread across classes.** For the best pipeline (jpeg → median → gaussian) on PGD, every class has an F1 of 0.71 or higher. The weakest classes are 50 mph (0.71) and 40 mph (0.85).

### Pixle accuracy

- Every defended pipeline scores 95.7% to 100%, up from 60%.
- All five pipelines ending with the median filter get a perfect 100%.
- Stacking helps only slightly: averages are 97.4% for singles, 98.8% for pairs and 99.4% for triples.

### Trade-offs

| Goal | Pipeline | Clean | PGD | Pixle |
|---|---|---|---|---|
| Highest PGD accuracy | jpeg → median → gaussian | 98.3% | 87.0% | 98.3% |
| No loss on clean or Pixle | jpeg → gaussian → median | 100% | 80.0% | 100% |
| Cheapest strong option (2 steps) | jpeg → gaussian | 98.3% | 74.8% | 98.3% |

Choosing between the top two means trading 7 points of PGD accuracy (8 images) for 1.7 points each of clean and Pixle accuracy (2 images each).

### Limits of this analysis

- **Small sample.** One image is 0.87 points, so gaps of 1 to 3 images (most clean and Pixle differences) are not meaningful on their own. The large PGD gaps between singles, pairs and triples are.
- **The PGD attack did not know about the defences.** These images were crafted against the undefended model. An attacker targeting the full pipeline would likely do better, so treat the PGD figures as an upper bound.
- **Settings were fixed, not tuned,** and this ranking was made on the test set. It shows what happened here; it is not a properly selected defence.
- **The loss column in `summary.csv` is not comparable across pipelines.** Any chain with Gaussian smoothing outputs vote shares, not model probabilities. Use accuracy and F1 for comparisons.
