# LightGCN Reproduction: A Scaled-Down Study on MovieLens

**Author:** Manish Chapagain
**Date:** November 2026
**Reference Paper:** He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.* SIGIR 2020.

---

## 1. Overview

This repository is a **scaled-down reproduction** of **LightGCN** (He et al., SIGIR 2020), a simplified Graph Convolutional Network for recommendation that removes feature transformation and nonlinear activation.

The paper claims LightGCN outperforms Matrix Factorization (BPR-MF) and NGCF on top-K metrics (Recall@20, NDCG@20), and that more layers (up to ~3-4) help. **Goal:** test these claims in a small, resource-constrained setting on MovieLens-small, and document honestly where this setup differs from the paper.

**Headline results** (100 epochs, 3 train/test splits, mean over splits):
- With **tuned hyperparameters**, LightGCN-1L is modestly better than BPR-MF: NDCG@20 +6.6% and Recall@20 +1.8%, higher in all 3 splits. The Recall gap is within noise; the NDCG gap is clearer but rests on only 3 splits.
- With **default hyperparameters** (lr = 1e-3), the two are indistinguishable. The conclusion therefore depends on tuning.
- LightGCN **trains much faster** than BPR-MF (e.g. +14% Recall@20 at epoch 20 with tuned settings).
- The paper's **layer-depth trend is not reproduced**: 1 layer is best, 3 layers worst.

---

## 2. Original Paper vs. This Reproduction

| Aspect | Original Paper | This Reproduction |
| :--- | :--- | :--- |
| **Datasets** | Gowalla, Yelp2018, Amazon-Book | MovieLens-small (ml-latest-small) |
| **Feedback** | Native implicit | Explicit ratings converted to implicit (rating >= 4) |
| **Scale** | ~29k users, ~41k items (Gowalla) | 609 users, 6,298 items |
| **Split** | Per-user random | Per-user 80/20 random; a different split per seed (2020, 2021, 2022) |
| **Negative sampling** | 1 negative per positive | On-the-fly uniform random sampling |
| **Regularisation** | L2 on batch embeddings (1e-4) | Adam weight_decay on all parameters |
| **Embedding dim / batch** | 64 / 2048 | 64 / 2048 |
| **Learning rate** | 1e-3 | Default 1e-3; tuned: 3e-3 (BPR-MF), 1e-2 (LightGCN) |
| **Layers** | 1 to 4 | 1, 2, 3 |
| **Epochs** | ~1000 | 100 |
| **Seeds** | - | 3 (vary initialisation, sampling and the split) |

This is an *adaptation*, not an exact replication.

---

## 3. Repository Structure

```text
lightgcn-repro/
├── data/ml-latest-small/         # MovieLens dataset (not tracked in git)
├── src/
│   ├── data.py                   # Loading, implicit conversion, normalized adjacency
│   ├── models.py                 # BPR-MF and LightGCN
│   ├── train.py                  # Single-run training/evaluation (20 epochs)
│   ├── run_experiments.py        # Multi-seed experiments with curves
│   └── tune.py                   # Hyperparameter search on a validation split
├── results_varysplit/            # Default hyperparameters (curves.csv, summary.csv, curves.png)
├── results_tuned/                # Tuned hyperparameters
├── results_tuning2/              # Tuning grid output
├── requirements.txt
└── README.md
```

---

## 4. Methodology

1. Keep interactions with `rating >= 4.0`; map IDs to contiguous indices.
2. Per-user 80/20 random split. The graph is built from **training edges only**, so test interactions do not leak into propagation.
3. Normalized adjacency `A_norm = D^{-1/2} A D^{-1/2}`, `A = [[0, R], [R^T, 0]]`.
4. **BPR-MF:** dot product of 64-d user/item embeddings. **LightGCN:** propagate embeddings through K layers (`E^(k+1) = A_norm E^(k)`), average layers 0..K, dot-product scoring. Both use BPR loss `-log sigmoid(s_pos - s_neg)`, so propagation is the only difference between them.
5. **Evaluation:** Recall@20 and NDCG@20 per user over all non-training items, averaged over users. Results are reported at the **final epoch**; the best epoch is never selected on the test set.
6. **Hyperparameter tuning:** 10% of each user's training items were held out as a validation set (seed 2020, test set untouched). Learning rate and Adam weight decay were searched for BPR-MF and LightGCN-1L over two overlapping grids (lr in {1e-3, 3e-3, 1e-2, 3e-2}, weight decay in {0, 1e-6, 1e-5, 1e-4, 1e-3}; 16 unique configs per model, 100 epochs, final-epoch validation Recall@20). Best: BPR-MF lr = 3e-3, wd = 1e-5; LightGCN-1L lr = 1e-2, wd = 1e-5. The LightGCN-1L setting was reused for 2 and 3 layers.

---

## 5. Results (epoch 100, 3 splits/seeds, mean ± std)

**Tuned hyperparameters (main result):**

| Model | Recall@20 | NDCG@20 |
| :--- | :--- | :--- |
| BPR-MF | 0.2333 ± 0.0064 | 0.2154 ± 0.0078 |
| **LightGCN-1L** | **0.2376 ± 0.0112** | **0.2296 ± 0.0114** |
| LightGCN-2L | 0.2240 ± 0.0075 | 0.2174 ± 0.0115 |
| LightGCN-3L | 0.2099 ± 0.0074 | 0.2050 ± 0.0113 |

**Default hyperparameters (lr = 1e-3, wd = 1e-5):**

| Model | Recall@20 | NDCG@20 |
| :--- | :--- | :--- |
| BPR-MF | 0.2141 ± 0.0044 | 0.2023 ± 0.0058 |
| LightGCN-1L | 0.2118 ± 0.0046 | 0.2056 ± 0.0065 |
| LightGCN-2L | 0.2026 ± 0.0059 | 0.1980 ± 0.0074 |
| LightGCN-3L | 0.1958 ± 0.0062 | 0.1911 ± 0.0086 |

**Paired difference vs. BPR-MF (tuned, same split and seed):**

| Model | ΔRecall@20 (rel.) | Splits won | p* | ΔNDCG@20 (rel.) | Splits won | p* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGCN-1L | +0.0043 (+1.8%) | 3/3 | 0.28 | +0.0142 (+6.6%) | 3/3 | 0.023 |
| LightGCN-2L | -0.0092 (-4.0%) | 0/3 | 0.07 | +0.0020 (+0.9%) | 2/3 | 0.68 |
| LightGCN-3L | -0.0234 (-10.0%) | 0/3 | 0.03 | -0.0104 (-4.8%) | 0/3 | 0.14 |

*Paired t-test with n = 3, not corrected for multiple comparisons; treat as a rough indication only.

**Recall@20 during training (tuned, mean over 3 splits):**

| Model | Epoch 5 | Epoch 10 | Epoch 20 | Epoch 50 | Epoch 100 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BPR-MF | 0.0643 | 0.1679 | 0.1962 | 0.2167 | 0.2333 |
| LightGCN-1L | 0.1960 | 0.2107 | 0.2232 | 0.2304 | 0.2375 |
| LightGCN-2L | 0.1908 | 0.2012 | 0.2139 | 0.2194 | 0.2240 |
| LightGCN-3L | 0.1848 | 0.1966 | 0.2027 | 0.2077 | 0.2099 |

Full curves: `results_tuned/curves.csv` and `results_tuned/curves.png`. Numbers on MovieLens-small are not comparable to the paper's results on other datasets (see Table 3 in He et al., 2020).

---

## 6. Findings & Discussion

**1. Tuning matters for both models.** Moving from lr = 1e-3 to the tuned settings raised BPR-MF's Recall@20 by 9% (0.2141 to 0.2333) and LightGCN-1L's by 12% (0.2118 to 0.2376). The default learning rate left both undertrained, and the comparison at default settings understated LightGCN-1L more.

**2. LightGCN-1L has a modest advantage once both models are tuned.** It is higher than BPR-MF in all 3 splits in both metrics, with a clearer NDCG@20 gap (+6.6%) than Recall@20 gap (+1.8%, within seed noise). With only 3 splits this is suggestive, not conclusive. Note also that the validation-set gap used for tuning (0.2385 vs 0.2245) was larger than the test gap (0.2376 vs 0.2333), as expected when configurations are selected on a single seed.

**3. LightGCN converges much faster.** At epoch 5, Recall@20 is 0.196 (LightGCN-1L) vs 0.064 (BPR-MF), +14% at epoch 20 and +6% at epoch 50, narrowing to +1.8% by epoch 100. With default settings the early gap is larger (+20% at epoch 20), so my first 20-epoch single-run result (+21%) mostly reflected convergence speed, not final quality. Recall is still rising for all models at epoch 100, and individual curves fluctuate by up to ~0.01 between evaluation points, so final-epoch numbers carry some noise.

**4. Layer-depth trend not reproduced.** The paper finds more layers help. Here 1L > 2L > 3L in Recall@20 in every split, under both default and tuned settings. Deeper models have higher training loss (about 0.17 for 3L vs 0.09 for 1L at epoch 100 in the tuned runs). Because 2L and 3L reused the 1L hyperparameters, they may be under-tuned; I have not tested whether separate tuning or much longer training would close the gap. Other factors I have not separated: stronger smoothing on this small graph, and the rating >= 4 conversion.

---

## 7. Limitations

1. **Dataset:** MovieLens-small is far smaller than the paper's datasets and has a different graph structure.
2. **Explicit-to-implicit conversion** (rating >= 4) differs from native implicit feedback.
3. **Training budget:** 100 epochs vs ~1000; no model is fully converged.
4. **Tuning:** small grid, a single seed, and one validation split; only BPR-MF and LightGCN-1L were tuned, and the 1L setting was reused for 2L and 3L. Adam weight decay differs from the paper's batch-level L2.
5. **Statistics:** 3 splits/seeds only; paired tests with n = 3 are weak and uncorrected for multiple comparisons.
6. **Negative sampling** is uniform and does not exclude a user's positive items.
7. **Cold items:** items whose interactions fall entirely in the test split have no training edges; this affects both models.

---

## 8. What I Learned

- How LightGCN's message passing works on the user-item bipartite graph, and how it differs from MF trained with the same loss.
- Building a reproducible pipeline (ingestion, filtering, indexing, adjacency construction, seeded experiments).
- That conclusions can change with training length and hyperparameter tuning: an initial single-seed, 20-epoch result (+21%) shrank to a modest gain after longer, tuned, multi-split runs.
- Tuning on a validation split rather than the test set, and reporting adaptations and uncertainty transparently.

---

## 9. How to Run

```bash
git clone <your-repo-url>
cd lightgcn-repro
python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

# Download https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# and extract into data/ so data/ml-latest-small/ratings.csv exists

python src/data.py                      # verify pipeline
python src/train.py                     # single run, 20 epochs

# Default hyperparameters, 3 splits
python src/run_experiments.py --epochs 100 --seeds 2020 2021 2022 --vary_split --out results_varysplit

# Hyperparameter search on a validation split
python src/tune.py --epochs 100 --lrs 0.003 0.01 0.03 --wds 0 1e-6 1e-5 --out results_tuning2

# Tuned hyperparameters, 3 splits
python src/run_experiments.py --epochs 100 --seeds 2020 2021 2022 --vary_split \
    --lr 0.01 --weight_decay 1e-5 --bpr_lr 0.003 --bpr_wd 1e-5 --out results_tuned
```

---

## 10. Future Work

- Train for 300+ epochs to see whether the models converge to the same level.
- Tune 2- and 3-layer LightGCN separately to test whether the layer trend depends on hyperparameters.
- Use more seeds/splits, and tune over several seeds, for firmer statistics.
- Run on a native implicit dataset (Gowalla or Yelp2018) with a GPU.
- Add knowledge-graph features (genres, tags) as a KG-aware extension.

---

## 11. Attribution

Student reproduction for educational purposes. All credit for LightGCN belongs to the original authors:

> He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.* SIGIR '20.

Dataset: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM TiiS.