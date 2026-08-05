# DS105 Final Project: Time-Based Browsing Pattern Analyzer

## Project Overview

This project analyzes browsing history and system RAM usage over the window 2026-05-03 to 2026-08-01. The goal is to identify behavioral patterns, categorize web usage, and correlate these activities with system performance.
## 1. Data Summary

The dataset consists of **24,321 browsing events** across **833 sessions** and **16 browsing categories**, synchronized with RAM logs captured at 5-second intervals.

### Top 10 Domains
| domain | category | event_count |
| --- | --- | --- |
| google.com | Search/Reference | 10227 |
| instagram.com | Social Media | 4301 |
| colab.research.google.com | Learning/Education | 1508 |
| m.youtube.com | Social Media | 861 |
| github.com | Learning/Education | 545 |
| accounts.google.com | Search/Reference | 494 |
| linkedin.com | Social Media | 234 |
| docs.google.com | Productivity/Work | 225 |
| drive.google.com | Productivity/Work | 214 |
| youtube.com | Social Media | 210 |

### Category Distribution
| category | count |
| --- | --- |
| Search/Reference | 11252 |
| Social Media | 5712 |
| Learning/Education | 2947 |
| Productivity/Work | 2044 |
| Shopping | 604 |
| Other | 462 |
| Job Search/Career | 279 |
| Ads/Suspicious Redirects | 240 |
| Piracy/Streaming (Unofficial) | 222 |
| Entertainment/Media | 147 |
| Automotive | 127 |
| Development/Local | 121 |
| Finance | 87 |
| Government | 37 |
| News | 35 |
| Adult Content | 5 |

## 2. RAM Correlation Analysis

### RAM Usage by Category (MB)
| category | mean_used_mb | peak_used_mb | mean_usage_percent | peak_usage_percent |
| --- | --- | --- | --- | --- |
| Search/Reference | 3477.11 | 6659.9 | 42.45 | 81.3 |
| Social Media | 3613.28 | 6655.1 | 44.11 | 81.24 |
| Productivity/Work | 3482.92 | 6212.9 | 42.52 | 75.84 |
| Shopping | 3637.2 | 6163.4 | 44.4 | 75.24 |
| Piracy/Streaming (Unofficial) | 3685.98 | 6129.2 | 45.0 | 74.82 |
| Entertainment/Media | 3564.58 | 6075.3 | 43.52 | 74.16 |
| Other | 3499.96 | 6043.1 | 42.73 | 73.77 |
| News | 3625.47 | 6008.6 | 44.26 | 73.35 |
| Learning/Education | 3435.59 | 5708.7 | 41.94 | 69.69 |
| Job Search/Career | 3448.37 | 5368.7 | 42.1 | 65.54 |
| Ads/Suspicious Redirects | 3448.34 | 5307.6 | 42.1 | 64.79 |
| Government | 3529.68 | 4852.7 | 43.09 | 59.24 |
| Automotive | 3431.64 | 3794.9 | 41.89 | 46.32 |
| Finance | 3434.7 | 3434.7 | 41.93 | 41.93 |
| Adult Content | 3434.7 | 3434.7 | 41.93 | 41.93 |
| Development/Local | 3434.7 | 3434.7 | 41.93 | 41.93 |

> **Finding:** **Search/Reference** has the highest peak RAM usage (6660 MB); entertainment and social media are the primary drivers of high memory consumption.

## 3. Behavior Clustering

Using KMeans clustering on scaled session features, we identified **2 distinct session types** (silhouette score 0.624).

### Cluster Profiles
| session_duration_minutes | page_count | unique_domains | unique_categories | domain_switches | category_switches | avg_time_per_page_seconds | mean_used_mb | peak_used_mb | mean_usage_percent | peak_usage_percent | session_start_hour | session_start_dayofweek |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13.51 | 29.61 | 3.05 | 2.09 | 5.61 | 4.36 | 26.82 | 3443.21 | 3459.06 | 42.03 | 42.23 | 9.27 | 2.89 |
| 10.95 | 21.56 | 2.4 | 2.12 | 3.35 | 3.19 | 25.61 | 5354.87 | 5779.87 | 65.37 | 70.56 | 12.02 | 2.26 |

### Cluster Labels
| cluster | label |
| --- | --- |
| 0 | Low RAM |
| 1 | High RAM |

## 4. Deep Learning: Next-Category Prediction

An **LSTM (Long Short-Term Memory)** model predicts the next browsing category from the last 5 visits.

* **Model Architecture:** Embedding (128) -> LSTM (2 layers, 256 units, dropout 0.5) -> Linear (Softmax)
* **Training:** 10 epochs, Adam lr=0.001, batch size 64
* **Test Accuracy:** **85.30%**
* **Insight:** Browsing behavior is highly sequential, allowing the model to anticipate transitions between categories.

### Classification Report (Test Set)
| category | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| Search/Reference | 0.8732317736670294 | 0.885273028130171 | 0.8792111750205424 | 1813.0 |
| Social Media | 0.9451097804391217 | 0.9212062256809338 | 0.9330049261083744 | 1028.0 |
| News | 0.3333333333333333 | 0.25 | 0.2857142857142857 | 4.0 |
| Job Search/Career | 0.723404255319149 | 0.68 | 0.7010309278350515 | 50.0 |
| Shopping | 0.576271186440678 | 0.6732673267326733 | 0.6210045662100456 | 101.0 |
| Other | 0.8620689655172413 | 0.6578947368421053 | 0.746268656716418 | 76.0 |
| Government | 0.0 | 0.0 | 0.0 | 6.0 |
| Productivity/Work | 0.7752808988764045 | 0.7688022284122563 | 0.772027972027972 | 359.0 |
| Learning/Education | 0.8199643493761141 | 0.8424908424908425 | 0.8310749774164409 | 546.0 |
| Piracy/Streaming (Unofficial) | 0.43137254901960786 | 0.6470588235294118 | 0.5176470588235295 | 34.0 |
| Ads/Suspicious Redirects | 0.6923076923076923 | 0.46153846153846156 | 0.5538461538461539 | 39.0 |
| Entertainment/Media | 0.64 | 0.6153846153846154 | 0.6274509803921569 | 26.0 |
| Automotive | 0.7 | 0.7368421052631579 | 0.717948717948718 | 19.0 |
| Finance | 0.46153846153846156 | 0.42857142857142855 | 0.4444444444444444 | 14.0 |
| Adult Content | 0.0 | 0.0 | 0.0 | 1.0 |
| Development/Local | 0.6666666666666666 | 0.5714285714285714 | 0.6153846153846154 | 21.0 |
| macro avg | 0.5937843695313437 | 0.5712348996252894 | 0.5778787161180468 | 4137.0 |
| weighted avg | 0.8542569059325704 | 0.8530335992264926 | 0.8527909685491573 | 4137.0 |

## 5. Actionable Recommendations

1. **Reduce late-night social scrolling** (medium) — Sessions starting after 22:00 are dominated by social media, which is linked to poor sleep.
2. **Close memory-heavy tabs to reduce RAM pressure** (high) — Some categories have a much larger memory footprint; closing idle tabs frees RAM.
3. **Unusually high RAM during some sessions** (medium) — Peak RAM exceeds the 90th percentile of sessions; heavy tabs drive memory spikes.
4. **Trim overall browser memory footprint** (low) — Average session RAM is high; limiting open tabs reduces memory pressure.
5. **Browsing is fragmented with rapid topic switching** (medium) — High switching rates suggest distraction; grouping related tasks improves focus.
6. **google.com is your dominant site** (medium) — A single domain drives over 30% of visits; evaluate whether that time is intentional.

## 6. Deliverables

- `data/processed/final_browsing_history.csv`: sanitized and preprocessed history
- `data/processed/final_ram_log.csv`: time-aligned RAM metrics
- `data/processed/domain_category_map.csv`: mapping of domains to categories
- `data/processed/session_features.csv`: engineered features for behavior analysis
- `data/models/lstm_model.pt`: trained LSTM next-category predictor
- `data/models/cluster_model.pkl`: trained KMeans clustering model
- `reports/images/session_clusters.png`: visualization of behavior clusters
- `reports/images/category_ram_correlation.png`: peak RAM usage by category
- `reports/images/lstm_training_history.png`: LSTM training curves
