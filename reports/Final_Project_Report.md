# DS105 Final Project: Time-Based Browsing Pattern Analyzer

## Project Overview
This project analyzes user browsing history and RAM usage over a 6-day period (July 27, 2026 – August 1, 2026). The goal is to identify behavioral patterns, categorize web usage, and correlate these activities with system performance.

## 1. Data Summary
The dataset consists of **24,320 browsing events** and synchronized RAM logs captured at 5-second intervals.

### Top 10 Domains
| Domain | Visit Count |
| :--- | :--- |
| google.com | 10,227 |
| instagram.com | 4,301 |
| colab.research.google.com | 1,508 |
| m.youtube.com | 861 |
| github.com | 545 |
| accounts.google.com | 494 |
| linkedin.com | 234 |
| docs.google.com | 225 |
| drive.google.com | 214 |
| youtube.com | 210 |

### Category Distribution
| Category | Count |
| :--- | :--- |
| Search/Reference | 13,631 |
| Social Media | 5,665 |
| Shopping | 307 |
| Learning/Education | 248 |
| Entertainment/Media | 57 |
| Productivity/Work | 28 |
| News | 14 |
| Other | 4,371 |

---

## 2. RAM Correlation Analysis
We analyzed the correlation between browsing categories and browser RAM consumption.

### RAM Usage by Category (MB)
| Category | Mean RAM (MB) | Peak RAM (MB) |
| :--- | :--- | :--- |
| **Entertainment/Media** | **1,237.31** | **1,353.54** |
| **Social Media** | **900.46** | **1,138.05** |
| Shopping | 768.02 | 913.65 |
| Productivity/Work | 653.62 | 793.45 |
| Learning/Education | 642.39 | 795.87 |
| Search/Reference | 480.53 | 836.73 |

> **Finding:** Entertainment and Social Media categories are the primary drivers of high memory usage, with Entertainment averaging over 1.2 GB per session.

---

## 3. Behavior Clustering
Using KMeans clustering, we identified 4 distinct session types based on duration, event count, and RAM usage.

| Cluster | Avg Duration (sec) | Avg Event Count | Mean RAM (MB) | Primary Behavior |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 3,280 | 159 | 527.92 | Quick Research |
| **1** | **4,706** | **228** | **796.19** | **High-RAM Social Media** |
| 2 | 3,648 | 184 | 627.66 | General Browsing |
| 3 | 11,628 | 553 | 566.67 | Deep Research/Work |

---

## 4. Deep Learning: Next-Category Prediction
We implemented an **LSTM (Long Short-Term Memory)** model to predict the next browsing category based on a sequence of the last 5 visits.

*   **Model Architecture:** Embedding Layer -> LSTM (32 units) -> Dense (Softmax)
*   **Performance:** The model achieved a **Test Accuracy of 88.23%**.
*   **Insight:** Browsing behavior is highly sequential, allowing the model to accurately anticipate transitions between research and social media.

---

## 5. Actionable Recommendations
1.  **Optimize High-RAM Categories:** Limit the number of open tabs for **Entertainment** and **Social Media** sites, as they consume 2-3x more RAM than research sites.
2.  **Session Management:** Sessions in Cluster 3 exceed 3 hours. Use a session timer to break long browsing periods, which helps in both digital wellbeing and clearing system cache.
3.  **Tab Hygiene:** Since **Search/Reference** accounts for over 50% of activity, use a tab manager to consolidate research links and reduce the browser's memory footprint.
4.  **Performance Balancing:** Avoid streaming media (Entertainment category) while working on memory-intensive platforms like Google Colab to prevent system slowdowns.
5.  **Predictive Intervention:** The LSTM model can be integrated into a digital wellbeing tool to warn users when they are likely to transition into a "distraction loop" (e.g., moving from Search to Social Media).

---

## 6. Deliverables
- `final_browsing_history.csv`: Sanitized and preprocessed history.
- `final_ram_log.csv`: Time-aligned RAM metrics.
- `domain_category_map.csv`: Mapping of 655 domains to categories.
- `session_features.csv`: Engineered features for behavior analysis.
- `session_clusters.png`: Visualization of behavior clusters.
- `category_ram_correlation.png`: RAM usage boxplots.
- `deep_learning.py`: LSTM implementation code.
