# Time-Based Browsing Pattern Analyzer

This project is a comprehensive analyzer for browser history and system RAM usage, developed for the DS105 Final Project.

## Project Structure
- `preprocess_data.py`: Cleans raw history and extracts domains.
- `generate_map.py`: Creates the domain-to-category mapping.
- `enhance_data.py`: Synchronizes history with RAM logs and adds browser-specific metrics.
- `sessionization.py`: Groups browsing events into sessions based on time gaps.
- `analysis.py`: Performs KMeans clustering and RAM correlation analysis.
- `deep_learning.py`: Implements an LSTM model for next-category prediction.
- `generate_summary.py`: Extracts key metrics for reporting.

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
   ```
2. Place your raw data in the `upload/` directory:
   - `browsing_history_last5days.csv`
   - `ram_log_last5days.csv`
3. Run the pipeline:
   ```bash
   python3 preprocess_data.py
   python3 generate_map.py
   python3 enhance_data.py
   python3 sessionization.py
   python3 analysis.py
   python3 deep_learning.py
   python3 generate_summary.py
   ```

## Key Findings
- **High Accuracy:** The LSTM model predicts the next browsing category with **88.2% accuracy**.
- **Performance Impact:** Entertainment and Social Media categories consume significantly more RAM (up to 1.3GB peak) compared to productivity tasks.
- **Behavioral Clusters:** Identified 4 distinct types of user sessions, ranging from quick research to intensive long-duration browsing.

## Deliverables
- Final Report: `Final_Project_Report.md`
- Visualizations: `session_clusters.png`, `category_ram_correlation.png`, `lstm_training_history.png`
- Processed Data: `final_browsing_history.csv`, `final_ram_log.csv`, `domain_category_map.csv`
