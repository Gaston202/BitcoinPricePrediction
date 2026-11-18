# Literature Review — ML Models for Bitcoin Direction Prediction

**Purpose.** Justify the candidate models, features and evaluation protocol for the rewritten modeling
phase (`notebooks/05`–`07`). The scope is papers that **classify the direction** of Bitcoin's next price move
from **price/volume/technical features** at daily or intraday horizons, which is the setup closest to ours
(3-class next-day direction, θ = 1 %, 1D Binance OHLCV).

**Method.** Searched with [`paper-search-mcp`](https://github.com/openags/paper-search-mcp) (arXiv,
OpenAlex, Semantic Scholar) on 2026-11-18 with queries such as *"bitcoin price direction classification
machine learning technical indicators"*. Price-regression papers (RMSE/MAPE on the price level) were excluded
because their metrics don't transfer to a classification task.

**Evidence level.** Each row is marked by how much of the paper was actually read:

- **F**: full text read (PDF open access).
- **A**: abstract only.
- **M**: metadata only (title, venue, keywords). The publisher full text wasn't reachable, so methodological details aren't verified.

## 1. Paper table

| # | Paper | Ev. | Models | Features | Horizon / label | Split / validation | Reported result | Leakage / validity flag |
|---|---|---|---|---|---|---|---|---|
| 1 | Akyıldırım, Göncü & Şensoy (2021), *Annals of OR* 297 | F | LR, SVM, ANN, RF | Past returns + technical indicators | Daily and 5/15/30-min; binary up/down | Chronological in-sample / out-of-sample; C tuned by CV on in-sample only | 55–65 % average accuracy; **SVM** best and most consistent | Low. Clean design, 12 coins |
| 2 | Sebastião & Godinho (2021), *Financial Innovation* 7:3 | F | Linear models, RF, SVM (classification + regression), ensembles | Trading + network (on-chain) activity, day-of-week dummies | Daily; sign of next return | **Train 50 % / validation 25 % / test 25 %**, rolling window; test is a bear regime | 5 of 18 single models < 50 % in test; only a **5-model unanimity ensemble** profitable after 0.5 % costs | Low. Explicitly tests regime change between validation and test, like our docs/07 |
| 3 | Ortu et al. (2022), *Expert Syst. Appl.* 116804 (arXiv 2102.08189) | F | MLP, CNN, LSTM, Attention-LSTM | Technical (restricted) vs + trading + social (unrestricted) | Hourly and daily; binary | Train/val/test + bootstrap of test samples; grid search | Technical-only **51–55 %**; with trading + social features 67–84 % (hourly) | Medium. The large gain comes from non-price features |
| 4 | Fernandes & Cunha (2025), CBIC 2025 | F | XGBoost, SVM, LR, RF | Technical, fundamental (CoinMetrics), lags 1–20 days | 1/3/7/14/30 days; **3-class buy/hold/sell with θ = α·σ (volatility-scaled)** | TimeSeriesSplit (5), RandomizedSearchCV on weighted F1 | **XGBoost** best on accuracy and trading; macro-F1 reported | Low–medium. Closest label design to ours; 30-day horizon returns look optimistic |
| 5 | Jung et al. (2023), *CSSE* 46(2) | F | XGBoost, LightGBM + 4 others | 11 technical indicators + VADER sentiment (Reddit/LexisNexis) | Daily; label = close − open **of the same day** | Not chronological-walk-forward | 90.6 % accuracy, AUC 0.97 (XGBoost) | **High.** Label uses same-day close; indicators computed from the same candle can see it |
| 6 | Öktem & Tekerek (2025), *Politeknik Dergisi* 28(6) | F | Random Forest *Regressor* thresholded to direction | **Single feature:** Balance of Power (BOP) | Daily % change | Train/val/test, shuffle off | **99.2 %** accuracy | **High.** BOP = (close − open)/(high − low) on the same day as the target return, so it encodes the answer |
| 7 | Fadilah, Yustiana & Nugraha (2026), *JSIT* 5(3) | F | RF (class-weighted) | 106 technical features | 1h; binary via **triple-barrier labeling** | TimeSeries CV tuning | Acc. 58.7 %, but **AUC 0.51**, recall 99.5 % | Low leakage, but the model is near-useless (AUC ≈ 0.5), which accuracy hides |
| 8 | Carosia (2024), arXiv 2410.14532 | F | LinReg, SVR, XGBoost, GB, RF, MLP | Technical + Fear & Greed index | Daily movement (regression → signal) | Chronological | Beats buy-and-hold in backtest | Medium. Single test window |
| 9 | Jaquart, Dann & Weinhardt (2021), *J. Finance & Data Sci.* 7 | A | LR, RF, **GB**, FNN, **LSTM/GRU** | Technical, blockchain, sentiment, asset-based | 1–60 min; binary | Chronological | All beat random; **GB and RNNs best**; technical features most relevant | Low. Trading is negative after costs |
| 10 | Jaquart, Köpke & Weinhardt (2022), *J. Finance & Data Sci.* 8 | A | LR, RF, GB, LSTM, GRU, ensembles | Return-based features, 100 coins | Daily; binary (relative to market) | Chronological, out-of-sample | **52.9–54.1 %** accuracy; 57.5–59.5 % on the top-10 % most confident predictions | Low. The most credible benchmark range |
| 11 | Huang, Huang & Ni (2019), *J. Finance & Data Sci.* 5 | A | Classification tree | 124 technical indicators | Daily; **narrow return-range classes** | Out-of-sample | "Strong" out-of-sample power for narrow return ranges | Low–medium |
| 12 | Dimitriadou & Gregoriou (2023), *Entropy* 25(5) | A | **LR**, linear SVM, RF | 24 variables: BTC lags, other cryptos, FX, macro | Daily; binary | Chronological | **LR best, 66 %** | Medium. Macro variables, single period (2014–2019) |
| 13 | El Badaoui et al. (2023), IEEE SITA | A | LR, DT, RF, SVM, **XGBoost**, ANN | Technical indicators + their signals | Daily; binary | **TimeSeriesSplit CV + Optuna** | Test accuracy > 65 % | Medium. Same tuning recipe we plan to use |
| 14 | Falcon & Lyu (2021), *J. Student Research* | A | 8 classifiers incl. **SVM**, AdaBoost, XGBoost | Price trend (7/30-day reversal), liquidity, volatility, network | Daily; sign of next return | Chronological | **SVM best; boosted models worst** | Medium. Contradicts #4 and #9 on boosting |
| 15 | Valencia et al. (2019), *Entropy* 21(6) | A | MLP, SVM, RF | Market data + Twitter sentiment | Daily; binary | — | MLP best | Medium |
| 16 | Máté et al. (2024), *J. Int. Studies* 17(3) | A | ANN, CNN-LSTM, SVM, RF | 27 technical indicators (momentum, ROC, %K) | Daily | — | SVM 82 %, ANN 81 % | **High (suspected).** Far above the 52–56 % consensus with price-only features |
| 17 | Radmand et al. (2025), *Int. J. Intelligent Systems* | A | XGBoost (DE-tuned), GB, SVM, NB | 768 TA-Lib features → 15 via TLBO selection | Multi-day; 5-class signal label | — | **91–92 %** accuracy | **High (suspected).** Forward-averaged label + feature selection on 768 features; implausible |
| 18 | Chen, Li & Sun (2020), *J. Comput. Appl. Math.* 365 | M | LR, LDA, SVM, RF, XGBoost, LSTM (OpenAlex keywords) | Price, blockchain, macro | Daily vs 5-min | — | — | Unverified. Highly cited (516); worth reading if the PDF becomes available |
| 19 | Mallqui & Fernandes (2019), *Applied Soft Computing* 75 | M | ANN, SVM, ensembles | Price, blockchain, macro, attribute selection | Daily; direction | — | — | Unverified |
| 20 | Alonso-Monsalve et al. (2020), *Expert Syst. Appl.* 149 | M | CNN, hybrid CNN-LSTM, MLP, RBF | Technical indicators | 1-min; trend classification | — | — | Unverified. Deep learning, intraday |

**Methodological reference (not Bitcoin):** Mesfin (2026), arXiv 2605.17724 [F]. It compares LSTM and gradient
boosting for direction on 5-min futures under **expanding-window walk-forward with permutation tests**.
No model beat the base rate (p = 0.135 for the best GB). It's a useful template for the significance test we
should add, and a reminder that a null result is a legitimate finding.

## 2. Which models the literature uses

| Model family | Papers | Verdict for our project |
|---|---|---|
| Logistic regression | 1, 8, 9, 10, 12, 13 | **Keep.** Best in #12, strong baseline everywhere |
| Random forest | 1, 2, 4, 7, 9, 10, 12, 13, 15, 16 | **Keep.** Most common model |
| Gradient boosting (sklearn GB / **XGBoost** / **LightGBM**) | 4, 5, 8, 9, 10, 13, 14, 17 | **Add XGBoost + LightGBM.** Best in #4, #9 and #13, worst in #14, so the evidence is mixed |
| SVM (RBF / linear) | 1, 2, 4, 12, 13, 14, 15, 16 | **Add.** Best in #1 and #14, the two cleanest daily studies |
| kNN | Chevallier et al. 2021 (regression) | **Add** as a cheap, different-bias baseline for the ensemble |
| Ensembles (voting / stacking / unanimity) | 2, 10, and Gyamerah 2019 (regression) | **Add soft voting + stacking.** #2's unanimity ensemble was the only profitable one in a bear test |
| MLP / ANN | 1, 13, 15, 16 | Optional (sklearn `MLPClassifier`, no new dependency) |
| LSTM / GRU / CNN / Transformer | 3, 9, 10, 20 | **Future work.** #9/#10 show RNNs competitive only with large intraday or multi-coin datasets; with ~2k daily rows, #3's restricted DL model gets 51–55 % |

## 3. Takeaways that shape our modeling

1. **Realistic ceiling.** The cleanest studies (#1, #3-restricted, #9, #10) put next-day direction accuracy
   at **52–56 %** from price features. Any result in the 80–99 % range (#5, #6, #16, #17) comes with a label that
   overlaps the features or with unverified selection on the full sample. This matches our own Boruta finding
   (no direction signal in OHLCV features) and the docs/07 result. Our success criterion of "a few points above
   persistence" is in line with the honest literature.
2. **There's no consistent winner.** SVM wins in #1 and #14, boosting in #4, #9 and #13, LR in #12, ensembles in #2.
   That justifies comparing **all** of them under one protocol rather than picking one from the literature.
3. **Volatility-scaled labels exist in the literature (#4)**, and triple-barrier labeling (#7) is a close
   relative. Both fit our docs/02 observation that a fixed θ = 1 % means different things in calm and wild
   regimes. We keep θ = 1 % (agreed decision) and note this as the main future-work lever.
4. **Selective prediction helps (#10).** Accuracy rises about 5 points on the top-10 % most confident
   predictions. That's compatible with our class-weight calibration variant and worth reporting as a coverage-vs-
   macro-F1 curve.
5. **Regime robustness is rarely tested.** Only #2 deliberately tests on a different regime than validation, and
   most single models fail there. Our 4-window walk-forward is stricter than almost all of the reviewed papers.
6. **Report AUC, not just accuracy (#7).** A model can show 58 % accuracy with AUC 0.51. Our criterion already
   requires macro-F1 **and** ROC-AUC.
7. **Indicator set for the feature ablation**, taken from #5, #13, #16, #17 and #7: RSI, MACD (line, signal,
   histogram), stochastic %K/%D, Bollinger %B and bandwidth, ROC, momentum, Williams %R, CCI, OBV, MFI and
   **lagged** BOP. Every indicator uses data up to `close[t]` only, which is the exact leakage mistake in #5 and #6.

## 4. Limitations of this review

- Paywalled full texts (#18–#20) and several abstracts-only rows weren't verified in detail. The table records
  claims, not replications.
- Only papers the search tools returned were included. It's not a systematic review (no PRISMA protocol).
