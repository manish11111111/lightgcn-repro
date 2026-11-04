# LightGCN Reproduction on MovieLens

A scaled-down reproduction of **LightGCN** (He et al., SIGIR 2020) on MovieLens-small, compared against a BPR-MF baseline.

**Author:** Manish Chapagain | **Date:** November 2026

**Paper:** He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.* SIGIR 2020.

---

## Overview

LightGCN is a graph convolution network for recommendation that drops feature transformation and nonlinear activation, keeping only neighbourhood aggregation over the user-item graph. The paper reports that it beats matrix factorization (BPR-MF) and NGCF on Recall@20 and NDCG@20, and that stacking up to 3-4 layers helps.

I wanted to see how much of this holds in a small setting: one small dataset, 100 epochs, and a laptop. This repo is an adaptation of the paper's setup, not an exact replication, and the differences are listed below.

**Summary of results** (100 epochs, 3 train/test splits, mean over splits):

- With tuned hyperparameters, LightGCN-1L is modestly better than BPR-MF: NDCG@20 +6.6% and Recall@20 +1.8%, higher in all 3 splits. The Recall gap is within noise. The NDCG gap is clearer, but 3 splits is not many.
- With default hyperparameters (lr = 1e-3) the two models are indistinguishable, so the conclusion depends on tuning.
- LightGCN trains much faster. With tuned settings its Recall@20 is 14% higher at epoch 20.
- The paper's layer-depth trend did not reproduce. Here 1 layer is best and 3 layers is worst.

---

## Original Paper vs. This Reproduction

| Aspect | Original paper | This reproduction |
| :--- | :--- | :--- |
| Datasets | Gowalla, Yelp2018, Amazon-Book | MovieLens-small (ml-latest-small) |
| Feedback | Native implicit | Explicit ratings converted to implicit (rating >= 4) |
| Scale | ~29k users, ~41k items (Gowalla) | 609 users, 6,298 items |
| Split | Per-user random | Per-user 80/20 random, a different split per seed (2020, 2021, 2022) |
| Negative sampling | 1 negative per positive | On-the-fly uniform random sampling |
| Regularisation | L2 on batch embeddings (1e-4) | Adam weight decay on all parameters |
| Embedding dim / batch size | 64 / 2048 | 64 / 2048 |
| Learning rate | 1e-3 | 1e-3 by default; tuned to 3e-3 (BPR-MF) and 1e-2 (LightGCN) |
| Layers | 1 to 4 | 1, 2, 3 |
| Epochs | ~1000 | 100 |
| Seeds | n/a | 3 (each changes initialisation, sampling and the split) |

---

## Repository Structure

```text
lightgcn-repro/
├── data/ml-latest-small/      # MovieLens dataset (not tracked in git)
├── src/
│   ├── data.py                # loading, implicit conversion, normalized adjacency
│   ├── models.py              # BPR-MF and LightGCN
│   ├── train.py               # single run, 20 epochs
│   ├── run_experiments.py     # multi-seed experiments with curves
│   └── tune.py                # hyperparameter search on a validation split
├── results_varysplit/         # default hyperparameters (curves.csv, summary.csv, curves.png)
├── results_tuned/             # tuned hyperparameters
├── results_tuning2/           # tuning grid output
├── requirements.txt
└── README.md
```

---

## Method

1. Keep interactions with `rating >= 4.0` and map user/item IDs to contiguous indices.
2. Split each user's interactions 80/20 at random. The graph is built from training edges only, so test interactions never leak into propagation.
3. Build the normalized adjacency `A_norm = D^{-1/2} A D^{-1/2}`, where `A = [[0, R], [R^T, 0]]` and `R` is the user-item interaction matrix.
4. **BPR-MF** scores a user-item pair with the dot product of 64-d embeddings. **LightGCN** propagates the embeddings through K layers (`E^(k+1) = A_norm E^(k)`), averages layers 0 to K, and scores with the dot product. Both are trained with BPR loss, `-log sigmoid(s_pos - s_neg)`, so graph propagation is the only difference between them.
5. **Evaluation:** Recall@20 and NDCG@20 per user, ranking all items the user has not trained on, then averaged over users. Results are taken at the final epoch. I never pick the best epoch using the test set.
6. **Tuning:** I held out 10% of each user's training items as a validation set (seed 2020, test set untouched). Learning rate and Adam weight decay were searched for BPR-MF and LightGCN-1L over two overlapping grids (lr in {1e-3, 3e-3, 1e-2, 3e-2}, weight decay in {0, 1e-6, 1e-5, 1e-4, 1e-3}; 16 unique configurations per model, 100 epochs, scored by final-epoch validation Recall@20). The best settings were lr = 3e-3, wd = 1e-5 for BPR-MF and lr = 1e-2, wd = 1e-5 for LightGCN-1L. The LightGCN-1L setting was reused for the 2- and 3-layer models.

---

## Results

Epoch 100, mean ± std over 3 splits.

**Tuned hyperparameters (main result)**

| Model | Recall@20 | NDCG@20 |
| :--- | :--- | :--- |
| BPR-MF | 0.2333 ± 0.0064 | 0.2154 ± 0.0078 |
| LightGCN-1L | 0.2376 ± 0.0112 | 0.2296 ± 0.0114 |
| LightGCN-2L | 0.2240 ± 0.0075 | 0.2174 ± 0.0115 |
| LightGCN-3L | 0.2099 ± 0.0074 | 0.2050 ± 0.0113 |

**Default hyperparameters (lr = 1e-3, wd = 1e-5)**

| Model | Recall@20 | NDCG@20 |
| :--- | :--- | :--- |
| BPR-MF | 0.2141 ± 0.0044 | 0.2023 ± 0.0058 |
| LightGCN-1L | 0.2118 ± 0.0046 | 0.2056 ± 0.0065 |
| LightGCN-2L | 0.2026 ± 0.0059 | 0.1980 ± 0.0074 |
| LightGCN-3L | 0.1958 ± 0.0062 | 0.1911 ± 0.0086 |

**Paired difference vs. BPR-MF (tuned, same split and seed)**

| Model | ΔRecall@20 (rel.) | Splits won | p* | ΔNDCG@20 (rel.) | Splits won | p* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LightGCN-1L | +0.0043 (+1.8%) | 3/3 | 0.28 | +0.0142 (+6.6%) | 3/3 | 0.023 |
| LightGCN-2L | -0.0092 (-4.0%) | 0/3 | 0.07 | +0.0020 (+0.9%) | 2/3 | 0.68 |
| LightGCN-3L | -0.0234 (-10.0%) | 0/3 | 0.03 | -0.0104 (-4.8%) | 0/3 | 0.14 |

\*Paired t-test with n = 3, not corrected for multiple comparisons. Treat the p-values as a rough indication only.

**Recall@20 during training (tuned, mean over 3 splits)**

| Model | Epoch 5 | Epoch 10 | Epoch 20 | Epoch 50 | Epoch 100 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BPR-MF | 0.0643 | 0.1679 | 0.1962 | 0.2167 | 0.2333 |
| LightGCN-1L | 0.1960 | 0.2107 | 0.2232 | 0.2304 | 0.2376 |
| LightGCN-2L | 0.1908 | 0.2012 | 0.2139 | 0.2194 | 0.2240 |
| LightGCN-3L | 0.1848 | 0.1966 | 0.2027 | 0.2077 | 0.2099 |

Full curves are in `results_tuned/curves.csv` and `results_tuned/curves.png`. These numbers are for MovieLens-small and cannot be compared with the paper's results on other datasets (Table 3 in He et al., 2020).

---

## Discussion

**Tuning changed the picture for both models.** Moving from lr = 1e-3 to the tuned settings raised BPR-MF's Recall@20 by 9% (0.2141 to 0.2333) and LightGCN-1L's by 12% (0.2118 to 0.2376). The default learning rate left both undertrained, and it hurt LightGCN-1L slightly more, so the comparison at default settings made LightGCN look worse than it is.

**The advantage of LightGCN-1L is small.** Once both models are tuned, it beats BPR-MF in all 3 splits on both metrics. The NDCG@20 gap (+6.6%) is clearer than the Recall@20 gap (+1.8%), which is inside the seed-to-seed noise. With 3 splits I would call this suggestive, not conclusive. The gap on the validation set that I used for tuning (0.2385 vs 0.2245) was also larger than the gap on the test set (0.2376 vs 0.2333), which is what you expect when settings are picked on a single seed.

**Most of the early gain is convergence speed.** At epoch 5, LightGCN-1L reaches a Recall@20 of 0.196 against 0.064 for BPR-MF. The relative gap is +14% at epoch 20 and +6% at epoch 50, and it narrows to +1.8% by epoch 100. With default settings the early gap is even larger (+20% at epoch 20). My first quick run (20 epochs, one seed) showed +21%, and that number mostly reflected this speed difference, not final quality. Recall is still rising for every model at epoch 100, and single curves move by up to about 0.01 between evaluation points, so the final-epoch numbers carry some noise.

**The layer-depth trend did not reproduce.** The paper finds that more layers help. Here 1L > 2L > 3L in Recall@20 in every split, under both default and tuned settings. The deeper models also end with a higher training loss (about 0.17 for 3 layers vs 0.09 for 1 layer at epoch 100, tuned). The 2- and 3-layer models reused the 1-layer hyperparameters, so they may simply be under-tuned, and I have not tested whether separate tuning or much longer training closes the gap. Two other explanations I could not separate are stronger smoothing on this small graph and the effect of the rating >= 4 conversion.

---

## Limitations

- **Dataset.** MovieLens-small is far smaller than the paper's datasets and has a different graph structure. Turning ratings >= 4 into implicit feedback is also not the same as native implicit data.
- **Training budget.** 100 epochs instead of ~1000, and no model has fully converged.
- **Tuning.** A small grid, one seed, one validation split. Only BPR-MF and LightGCN-1L were tuned, and the 1L setting was reused for 2L and 3L. Adam weight decay also differs from the paper's batch-level L2.
- **Statistics.** Only 3 splits, so the paired tests are weak and uncorrected for multiple comparisons.
- **Smaller issues.** Negative sampling is uniform and does not exclude a user's own positives. Items whose interactions all fall in the test split have no training edges, which affects both models.

---

## What I Learned

- "Higher in 3 of 3 splits" can hide how small a win is. For Recall@20, LightGCN-1L beats BPR-MF by only +0.0008 on split 2021 and +0.0020 on split 2022, so most of the +0.0043 average comes from split 2020. The NDCG@20 gap is more even across splits, which is why it looks clearer. I only noticed this when I went back to the per-split rows in `curves.csv` instead of the averages.
- A lower training loss did not mean better recommendations. At epoch 100 BPR-MF has a lower BPR loss than LightGCN-1L (about 0.066 vs 0.093) but a slightly lower test Recall@20. Averaging over neighbours seems to act as a mild regulariser. With more layers the same smoothing goes too far: LightGCN-3L ends at a training loss of about 0.17, and on split 2020 its Recall@20 barely moved between epoch 25 (0.2155) and epoch 100 (0.2157).
- Evaluating only at the final epoch is noisy. On split 2021, LightGCN-1L went from 0.2346 at epoch 80 to 0.2247 at epoch 90 and back to 0.2320 at epoch 100. LightGCN-2L on the same split peaked around epoch 65-70 and its NDCG@20 then fell from 0.2164 to 0.2047. Choosing a different evaluation epoch could shift the small gaps in this repo.
- My first result (+21% after 20 epochs, one seed) mostly measured convergence speed. It only shrank to its real size after I trained longer, tuned on a validation split and used several splits.

---

## How to Run

```bash
git clone https://github.com/manish11111111/lightgcn-repro
cd lightgcn-repro
python -m venv venv
venv\Scripts\activate          # Windows (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

# Download https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# and extract into data/ so that data/ml-latest-small/ratings.csv exists

python src/data.py             # check that the data pipeline works
python src/train.py            # single run, 20 epochs

# Default hyperparameters, 3 splits
python src/run_experiments.py --epochs 100 --seeds 2020 2021 2022 --vary_split --out results_varysplit

# Hyperparameter search on a validation split
python src/tune.py --epochs 100 --lrs 0.003 0.01 0.03 --wds 0 1e-6 1e-5 --out results_tuning2

# Tuned hyperparameters, 3 splits
python src/run_experiments.py --epochs 100 --seeds 2020 2021 2022 --vary_split \
    --lr 0.01 --weight_decay 1e-5 --bpr_lr 0.003 --bpr_wd 1e-5 --out results_tuned
```

---

## Future Work

- Train for 300+ epochs to see whether the models end up at the same level.
- Tune the 2- and 3-layer models separately to check whether the layer trend depends on hyperparameters.
- Use more seeds and splits, and tune over several seeds.
- Run on a native implicit dataset (Gowalla or Yelp2018) with a GPU.
- Add knowledge-graph features (genres, tags) to the graph.

---

## Attribution

Student reproduction for educational purposes. All credit for LightGCN goes to the original authors:

> He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.* SIGIR '20.

Dataset: Harper, F. M. & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context.* ACM TiiS.
