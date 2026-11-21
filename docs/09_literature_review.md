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
| 9 | Jaquart, Dann & Weinhardt (2021), *J. Finance & Data Sci.* 7 | F | LR, RF, **GB (XGBoost)**, FNN, **LSTM, GRU** (Keras) | Technical, blockchain, sentiment (Google NL API), asset-based; RNNs see the last 120 minutes | 1/5/15/60 min; binary, **threshold = training-set median return** (balanced classes) | Chronological, Mar–Dec 2019: **train 5/9, validation 1/9 (tuning + early stopping), test 3/9**; standardisation fitted on train only | All beat random; **GB and RNNs best**; predictability rises with horizon | Low. Exemplary design. Long-short strategy +39 %/month before costs, **negative after costs** |
| 10 | Jaquart, Köpke & Weinhardt (2022), *J. Finance & Data Sci.* 8 | F | LR, RF, GB, TCN, LSTM, GRU, ensembles | **27 multi-period returns** (lags 1–20, 30–90 days), standardised on train | Daily; binary: does a coin **beat the cross-sectional median** of 100 coins tomorrow | **5 overlapping 800-day study periods**, each 500 train / 150 validation / 150 test days | **52.9–54.1 %** accuracy; 57.5–59.5 % on the top-10 % most confident predictions; LSTM/GRU ensemble Sharpe 3.2 after costs | Low. Multi-period testing ≈ our walk-forward. Relative (cross-sectional) target, not absolute BTC direction |
| 11 | Huang, Huang & Ni (2019), *J. Finance & Data Sci.* 5 | F | **CART** classification trees, 1,000 trees aggregated | **124 technical indicators** | Daily; **21 non-overlapping return intervals** for next-day return | Chronological: train Apr 2012–Jul 2016 (1,539 days), test Jul 2016–Dec 2017 (509 days) | "Strong" out-of-sample power for narrow return ranges; beats buy-and-hold | Medium. Single test window that is almost entirely the 2017 bull run |
| 12 | Dimitriadou & Gregoriou (2023), *Entropy* 25(5) | F | **LR**, linear SVM, RF | 24 variables: other cryptos, FX rates, interest rates, EPU index | **Weekly** (239 obs.); binary sign of BTC return | 80/20 split + **standard 5-fold CV** (not time-series CV); chronology of split and feature lags not stated | **LR best, 66 %** | **Medium–high.** Tiny sample; k-fold CV on time series; same-period crypto returns may be contemporaneous with the target |
| 13 | El Badaoui et al. (2023), IEEE SITA | A | LR, DT, RF, SVM, **XGBoost**, ANN | Technical indicators + their signals | Daily; binary | **TimeSeriesSplit CV + Optuna** | Test accuracy > 65 % | Medium. Same tuning recipe we plan to use |
| 14 | Falcon & Lyu (2021), *J. Student Research* | A | 8 classifiers incl. **SVM**, AdaBoost, XGBoost | Price trend (7/30-day reversal), liquidity, volatility, network | Daily; sign of next return | Chronological | **SVM best; boosted models worst** | Medium. Contradicts #4 and #9 on boosting |
| 15 | Valencia et al. (2019), *Entropy* 21(6) | A | MLP, SVM, RF | Market data + Twitter sentiment | Daily; binary | — | MLP best | Medium |
| 16 | Máté et al. (2024), *J. Int. Studies* 17(3) | A | ANN, CNN-LSTM, SVM, RF | 27 technical indicators (momentum, ROC, %K) | Daily | — | SVM 82 %, ANN 81 % | **High (suspected).** Far above the 52–56 % consensus with price-only features |
| 17 | Radmand et al. (2025), *Int. J. Intelligent Systems* | F | XGBoost tuned by Differential Evolution; GB, SVM, Naive Bayes as comparisons | 768 features (TA-Lib RSI/SMA/WMA/MOM/VAR over 3–90 days, on-chain, plus 6 stock indices) → 15 via **TLBO** metaheuristic selection | Multi-day; **5-class** (strong sell … strong buy) from the % change of the **average price over several days** | **Randomly shuffled 80/20 split** after labeling and feature extraction | **91–92 %** accuracy | **High (confirmed).** Shuffling with labels that span several days lets near-identical neighbouring days sit in train and test, which inflates accuracy |
| 18 | Chen, Li & Sun (2020), *J. Comput. Appl. Math.* 365 | F | Daily: **LR, LDA** (simple, against overfitting). 5-min: **RF, XGBoost, QDA, SVM, LSTM** | Daily: 15 high-dimensional features (network, trading, Google Trends, Baidu media, gold). 5-min: OHLCV only | Daily (Feb 2017–Feb 2019, ~730 days) and 5-min (Jul 2017–Jan 2018, ~50k); binary ±1 | **Chronological 75/25** | Daily: **LR/LDA 66 %**, beating complex ML (65.3 %). 5-min: ML best at **67.2 %** | Medium. Daily test ≈ the 2018 crash, so accuracy without a majority-class baseline is hard to judge. Key idea: **match model complexity to sample size** |
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

8. **Match model complexity to sample size (#18).** On ~730 daily rows, LR and LDA beat RF, XGBoost and SVM.
   Complex models only won on ~50k 5-minute rows. We have ~2k daily rows, so strong regularisation and
   simple baselines must stay in the comparison, and Optuna search spaces should favour shallow trees.
9. **The two confirmed leakage patterns are worth naming in the report:** (a) the feature and the label come
   from the same candle (#5, #6); (b) a **shuffled split** with labels that span several days (#17). Our pipeline
   avoids both, since labels use `close[t+1]` only and every split is chronological.
10. **Balanced labels from training-set quantiles (#9).** Setting the class threshold from training data only
    avoids majority-class bias without leakage. It's a cleaner alternative to class weights, and worth noting as an
    option alongside the volatility-scaled label.

## 4. Extension (2026-11-21): target design, intraday horizons and validation

**Why.** The next modeling iteration was agreed on 2026-11-21. It tests 1h, 4h and 1D horizons, binary and
volatility-scaled 3-class targets, and short-term (intraday, fast and candle) features. Features are
re-selected inside each walk-forward window. The models are still judged on ML metrics only. This extension
reviews what the literature says on those exact choices.

**Method.** Same tools (`paper-search-mcp`: arXiv, OpenAlex). Queries covered triple-barrier labeling,
cryptocurrency label methods, intraday and hourly Bitcoin direction, walk-forward and backtest-overfitting
methodology, and probability calibration for direction classifiers. #21 was read in its main sections: design,
features, selection and results. #22 was read for its abstract, design, labeling and conclusions.

### 4.1 Paper table (continued)

| # | Paper | Ev. | Models | Features | Horizon / label | Split / validation | Reported result | Leakage / validity flag |
|---|---|---|---|---|---|---|---|---|
| 21 | Bysik & Ślepaczuk (2026), arXiv 2606.00060 | F | **XGBoost**, LSTM, iTransformer | Three nested sets: OHLCV → + technical indicators → + EGARCH volatility features. **Indicators are computed causally inside each fold** | **1h BTCUSDT**, about 70k bars, 2018–2026; regression on the next-hour return, traded on the sign | **Rolling walk-forward: 12 months train / 3 months validation / 3 months test, 27 folds**; Optuna; selection by validation loss, information coefficient (IC) or information ratio; **circular block bootstrap** tests | Naive sign trading **loses heavily after 10 bp costs** (long-only −64 %/yr). A cost-aware filter (trade only if the forecast > λ × cost) restores profit (Sharpe ≈ 1.1). XGBoost is descriptively best, but **not significantly** better than the deep models. Technical indicators help only in some cases; EGARCH features are not robust | Low. Exemplary intraday protocol. The authors note the validation segment is reused for tuning and selection |
| 22 | Grądzki, Wójcik & Lessmann (2025), *Financial Innovation* 11:136 | F | ResNet-LSTM, Transformer, FEDformer, Autoformer, TSMixer, MLP, XGBoost and others | OHLCV-derived features on time bars vs information-driven bars (CUSUM filter, range, volume, dollar) | BTC/ETH tick data 2018–2023. **Next-bar vs triple-barrier label** (vertical barrier 24 bars, symmetric ±2.5 % / ±5 % barriers, **not** volatility-scaled) | **Quarterly expanding window**, 5 test quarters (Q2 2022–Q2 2023); validation for early stopping and model choice | **Triple barrier beats next-bar** on accuracy and on profit after costs. Next-bar labels "flip frequently, forcing the model to learn very small price movements". CUSUM + triple barrier is consistently profitable (Sharpe ≈ 2). Deep models beat classical ML on tick-sized data | Low–medium. Short test span (5 quarters) and many sampling × labeling × model configurations, so there is a multiple-testing risk. The barrier sizes were set by hand |
| 23 | Bieganowski & Ślepaczuk (2024), arXiv 2411.12753 | A | Supervised autoencoders | Fractionally differentiated features, noise augmentation | BTC, LTC, ETH 2016–2022; **triple-barrier labels** | Out-of-sample; Sharpe and information ratio | Triple barrier plus a moderate bottleneck and noise improves strategies; too much noise hurts | Medium (abstract only) |
| 24 | Fu, Kang, Hong & Kim (2024), *Mathematics* 12(5):780 | A | ML classifiers on genetic-algorithm-optimised labels | Pair-trading spread features | Crypto pairs; triple barrier with **barriers optimised by a genetic algorithm** for profit or minimum drawdown | Train 2017-11 → 2022-08, test 2022-09 → 2023-12 | +51 % profit (high-risk labels); −73 % max drawdown (low-risk labels) | Medium. Optimising the label definition itself is another way to overfit |
| 25 | El Youssefi et al. (2026), Springer LNNS | M | — | — | Crypto returns; **triple barrier across several timeframes** | — | — | Unverified |
| 26 | Kili, Raouyane, Rachdi & Bellafkih (2025), *Applied Sciences* 15(24):13204 | A | Several ML models; proposed AEDL labeling | Multi-scale features | 16 assets 2000–2025. **Head-to-head comparison of labeling methods** | "Rigorous out-of-sample"; Wilcoxon tests | Mean Sharpe by label method: **fixed horizon −0.29**, triple barrier −0.03, trend scanning 0.00, AEDL 0.48. Their labels output **calibrated probabilities** | Medium. Not crypto-specific, and the proposed method is complex. Its value here is the baseline comparison: **fixed-horizon labels rank last** |
| 27 | Wu, Wang, Su, Tang & Wu (2020), *Entropy* 22(10):1162 | A | LSTM, GRU and others | Price features, with **look-ahead-free normalisation** | Chinese indices and stocks; "continuous trend" labels vs next-period labels | Chronological | Trend labels beat next-period labels on classification metrics | Medium. Stocks, not crypto |
| 28 | Ma, Ventre & Polukarov (2022), ICAIF '22 | A | Self-supervised label denoising | — | Direction labels; **noise in labels** ("did the price go up because of noise or a signal?") | — | Denoised labels help learning | Medium. Supports moving away from raw next-bar sign labels |
| 29 | Shintate & Pichl (2019), *JRFM* 12(1):17 | A | Deep-learning random sampling method, LSTM, baselines | High-frequency price features | Intraday BTC (OkCoin); **3-class trend** | Chronological | Baselines are **biased by class imbalance**. The F-measure beats a uniform 3-outcome random process, but profit does **not** beat buy-and-hold | Low–medium. Honest null on trading; same imbalance problem we saw (raw argmax → "flat") |
| 30 | Akyıldırım, Çepni, Corbet & Uddin (2021), *Annals of OR* | M | ML classifiers | Futures order-book and price features | Intraday **Bitcoin futures mid-price direction** | — | — | Unverified |
| 31 | Bouri, Lau, Saeed, Wang & Zhao (2021), *Int. Rev. Fin. Analysis* 101784 | M | Functional time-series models | Intraday return curves | Intraday BTC | — | Reports intraday predictability and trading opportunities | Unverified |
| 32 | Jasiak & Zhong (2025), arXiv 2505.20508 | A | Functional PCA (FPCA), compared with ML and ARMA | Intraday return functions | **1h and 15-min BTC**; return sign at fixed times | Rolling | FPCA beats ML and ARMA on **return sign**; modelling heteroscedasticity improves interval forecasts | Low–medium |
| 33 | Mroziewicz & Ślepaczuk (2026), arXiv 2602.10785 | A | EMA trading rule | — | **1–60 min BTC** | Walk-forward with **81 train/test window-length combinations**; a "double out-of-sample" period run **once** | Results depend strongly on the walk-forward window length. 0.1 % fees; break-even at about 0.4 % | Low. Window length is itself a hyperparameter that can be overfitted |
| 34 | Michańków, Sakowski & Ślepaczuk (2023), arXiv 2309.15640 | A | LSTM, ARIMA-GARCH, momentum, contrarian | Prices | Daily 2004–2022, plus a **1h test for BTC** | Walk-forward | The LSTM at **1h beats the same model on daily data** | Medium |
| 35 | Stefaniuk & Ślepaczuk (2025), arXiv 2503.18096 | A | Informer | Prices | **5/15/30-min BTC**, 6 test periods | Walk-forward | **Loss function decides the outcome**: an RMSE-trained model gets worse at higher frequency, while a directional loss (GMADL) improves | Medium |
| 36 | Huang, Liu & Li (2026), Research Square preprint | A | LSTM, Transformer, **XGBoost** + a regime-aware decision threshold | — | Daily BTC 2014–2026; probability output | **8-fold walk-forward** | **XGBoost is the best-calibrated model (ECE 0.030)**. Regime-aware thresholds help the neural models but hurt XGBoost. A temperature-scaling experiment rejects calibration as the causal channel | Medium. Not peer reviewed |
| 37 | Kuznetsov et al. (2025), *Applied Sciences* 15(20):11145 | A | Neural network | Daily macro indicators + **order-book microstructure** | 11 crypto pairs, 12 months; **binary up/down only, confidence threshold decides whether to trade** | — | 82.7 % accuracy on the 12 % of hours traded | **Medium–high.** One year, order-book data, and accuracy far above the consensus. Useful only as an example of the *binary + confidence band* design |
| 38 | Lukito, Haryono & Santosa (2026), *IJACSA* 17(9) | A | XGBoost, **LightGBM**, BiLSTM, calibrated ensemble | Technical | Daily stock direction | **Walk-forward** | Uncalibrated **LightGBM ECE 0.205**, vs 0.035 after **temperature scaling** + confidence weighting | Medium. Stocks. Shows boosted-tree probabilities can be badly calibrated without a fix |
| 39 | Arian, Norouzi Mobarekeh & Seco (2024), *Knowledge-Based Systems* 112477 | M | — | — | Synthetic controlled market | Compares out-of-sample testing methods (walk-forward, k-fold, purged/combinatorial purged CV) for backtest overfitting | Reported to favour combinatorial purged CV over a single walk-forward path at limiting false discoveries | Unverified (metadata only) |
| 40 | Nazarkhani & Kahraman (2026), *Scientific Reports* | A | Pretrained visual encoders, LSTM, GRU, Transformer | Price charts | Gold futures; **balanced 3-class SELL/HOLD/BUY** | **Purged + embargoed CV** | Exploratory single-split accuracy 64 % → **44 %** under the confirmatory purged protocol (chance = 33 %). Sequence baselines are at chance | Low. A direct demonstration that the validation protocol, not the model, drives headline numbers |
| 41 | Tanrıkulu & Pabuçcu (2025), *Computational Economics* | A | RF, kNN, XGBoost, SVM, NB, ANN, LSTM | 12 technical indicators + Google Trends + tweet counts; continuous vs discretised ("trend") inputs | Daily BTC; 1-year vs 7-year samples | — | ANN/SVM best on continuous inputs; **discretising indicators** into trend signals improves accuracy; sample size matters | Medium |

### 4.2 Takeaways for the planned iteration

11. **Fixed-horizon next-bar labels rank last in the head-to-head comparisons (#22, #26, #27, #28).** They "flip
    frequently" and make the model chase moves smaller than the noise. Path-dependent labels (**triple barrier**:
    take-profit / stop-loss / time limit) and trend labels do better in all four. This supports our planned
    volatility-scaled 3-class target. It also suggests adding a **volatility-scaled triple-barrier target** as a
    third candidate, with the vertical barrier equal to the horizon. Two cautions:
    - #24 shows that **tuning barrier sizes to maximise profit overfits the label itself**. Barrier multipliers
      must be fixed in advance or chosen on inner validation only.
    - A triple-barrier label spans several bars, so the **embargo between train, validation and test must be at
      least the vertical barrier length** (#40). With a horizon-length vertical barrier, our existing embargo of
      `LABEL_HORIZON` bars is enough.
12. **Intraday horizons are worth testing, but costs dominate there (#21, #33).** At 1h, a naive sign strategy
    loses about 64 %/yr after 10 bp, and profit only appears when trades are filtered by forecast strength.
    For our ML-only phase, this means the 1h model has to be judged on **calibrated confidence**, not on raw
    accuracy. It also means a confidence band (only trade when P(up) is far from 0.5) is a natural output for
    the agent. #34 finds hourly LSTM beats daily. #35 finds the result depends on the loss function. There is
    almost **no 4h literature**, so 4h is a genuine gap that our study can fill.
13. **Clean intraday studies use many short walk-forward folds.** Examples: #21 has 27 quarterly folds, #22 five
    quarterly folds, #36 eight folds. Our 4 fixed calendar windows are fine for 1D (about 2.8k rows) and line up
    across horizons. For 1h and 4h, the data supports extra **quarterly rolling folds inside each window**
    without changing the windows. The window length itself is a hyperparameter (#33) and should be fixed
    up front, not tuned.
14. **Compute features causally inside each fold (#21)**, the same way we planned to re-select features per
    window. #27 adds that **normalisation must also be fitted on past data only**. Our pipeline already fits
    scalers on the training split.
15. **Test significance with a block bootstrap or permutation test, not a single number (#21, Mesfin 2026).**
    With 365 test days per window, a lift of 2–4 macro-F1 points over persistence may not be significant. A
    paired **circular block bootstrap** of model minus persistence per window is cheap and was already on our
    future-work list.
16. **Boosted-tree probabilities need checking before the agent uses them.** #38 measures an uncalibrated
    LightGBM ECE of 0.205 on stocks, while #36 finds XGBoost on daily BTC already well calibrated (0.030). Our
    plan covers this: Platt scaling on inner validation, with log-loss, Brier and **ECE** reported. Add a
    reliability diagram per horizon.
17. **Model choice: boosting stays the safe default at our data sizes.** At 1h (about 70k rows), XGBoost matches
    or beats LSTM and iTransformer, but no model is significantly better (#21). Deep models only pull ahead on
    tick-level data (#22). This confirms that keeping the 9 classical models and leaving deep learning as
    future work is reasonable.
18. **Validation protocol inflates or deflates headline numbers more than model choice does (#39, #40).** #40
    drops from 64 % to 44 % (chance 33 %) when switching to purged, embargoed CV. Our walk-forward with an
    embargo is already on the conservative side. Report it explicitly, alongside the per-window feature
    re-selection.

### 4.3 Changes to the agreed plan (all five accepted 2026-11-21)

| # | Suggestion | Based on | Effect on the plan |
|---|---|---|---|
| S1 | Add a **volatility-scaled triple-barrier target** (barriers = k × rolling σ, vertical barrier = horizon, k fixed or chosen on inner validation) at each horizon | #22, #23, #26 | One more target family (3 per horizon instead of 2) |
| S2 | Add a **circular block bootstrap** of each model's lift over persistence per window, reporting a p-value | #21, Mesfin 2026 | Small extra computation; turns "beats persistence" into "significantly beats persistence" |
| S3 | For 1h and 4h, also report **quarterly rolling folds** inside the 4 windows | #13 above (#21, #22, #36) | More robust intraday estimates; the 4 windows stay the primary comparison |
| S4 | Report **ECE + reliability diagrams** next to log-loss and Brier | #36, #38 | Reporting only |
| S5 | Keep the embargo at least equal to the label span for every target (already true for horizon-length labels) | #40 | No change in code if S1 uses a vertical barrier = horizon |

## 5. Limitations of this review

- Rows marked **A** or **M** weren't verified in detail. Alonso-Monsalve (#20) and Máté (#16) have legal
  open-access copies, but the PDFs weren't retrieved. Mallqui (#19), Borges & Neves and El Badaoui (#13) aren't
  open access. The table records claims, not replications.
- Only papers the search tools returned were included. It's not a systematic review (no PRISMA protocol).
- In the 2026-11-21 extension, only #21 and #22 were read in full. #25, #30, #31 and #39 are metadata only, and
  the claim attributed to #39 is unverified. #36 is a preprint, not peer reviewed. Semantic Scholar returned no
  results during that search session, so coverage relies on arXiv and OpenAlex.
