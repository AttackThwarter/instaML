[![مشاهده نسخه فارسی](https://img.shields.io/badge/-%D9%85%D8%B4%D8%A7%D9%87%D8%AF%D9%87%20%D9%86%D8%B3%D8%AE%D9%87%20%D9%81%D8%A7%D8%B1%D8%B3%DB%8C-8A2BE2?style=for-the-badge&logo=googletranslate&logoColor=white)](README_fa.md)


# Project Documentation: Instagram Data Analysis with ECLAT and Anomaly Detection
 
 
## 1. Project Overview
 
This project is a comprehensive analytical tool for extracted Instagram data (especially video posts). Its main goal is to uncover hidden patterns in user behavior and content, identify anomalous posts and users, and detect potential fraud signals using a combination of data mining and machine learning techniques.
 
This system is specifically designed for analyzing Persian content and identifying Iranian users. The results are presented through textual reports and visual dashboards.
 
---
 
## 2. Main Objectives
 
### Pattern Discovery:
 
Using the ECLAT algorithm, frequent patterns in three main areas are identified:
 
* **Hashtag Patterns:** Which hashtags and combinations are most frequently used together?
* **User Behavior Patterns:** What common traits do successful users share (e.g., active times, engagement rates)?
* **Temporal Patterns:** On which days and times do posts perform best?
 
### Anomaly Detection:
 
Using the Isolation Forest algorithm, outliers that differ from overall data behavior are identified:
 
* **Anomalous Posts:** Posts with significantly different likes, comments, or engagement rates.
* **Anomalous Users:** Users with suspicious activity patterns (e.g., sudden growth, low activity but high engagement).
* **Anomalous Content:** Posts with unusual content features (e.g., very short videos that went viral).
 
### Fraud Signal Detection:
 
Using predefined rules and thresholds, users with suspicious behaviors indicative of fraud (like buying likes or views) are flagged.
 
### Report Generation & Visualization:
 
Analysis results are presented as detailed textual reports and understandable graphic dashboards for better decision-making.
 
---
 
## 3. Project Structure
 
### `main.py`:
 
* The core engine of the project.
* Contains the `InstagramECLATAnalyzer` class, which handles data loading, preprocessing, algorithm execution, and output generation.
 
### `config.py`:
 
* Acts as the project settings panel.
* All parameters, file paths, column names, algorithm thresholds, and chart appearance settings are defined here, allowing easy customization without modifying the core code.
 
### `instagramData.xlsx`:
 
* The Excel input file containing raw extracted Instagram data.
* The project reads data from the sheet specified in `config.py`.
 
---
 
## 4. Methodology
 
### Step 1: Data Loading & Preprocessing
 
* Data is loaded from the Excel file defined in `config.py`.
* Numeric columns (like likes and comments) are converted to numeric types, and invalid values are replaced with zero.
* Posts with zero likes are excluded from analysis.
* **Nationality Detection:** User nationality is determined as "Iranian" or "International" based on Persian keywords (`iranian_keywords`) and alphabet detection in captions and hashtags.
* **Feature Engineering:** New features such as post hour, weekday, and like category (e.g., 'Viral', 'High', 'Medium') are created for deeper analysis.
 
### Step 2: Pattern Analysis with ECLAT
 
This algorithm finds frequent itemsets:
 
* **Hashtag Analysis:** Identifies hashtag combinations frequently used together.
* **User Analysis:** Profiles users based on features like "High/Medium/Low Activity" and "High/Medium/Low Engagement" to uncover common patterns.
* **Time-Based Analysis:** Discovers temporal patterns in posting (e.g., "Morning Post + Friday + High Performance").
 
### Step 3: Anomaly Detection with Isolation Forest
 
* An unsupervised algorithm to identify outliers.
* Run on three feature sets to detect anomalous posts, users, and content.
* Each item is assigned an "Anomaly Score" — the lower the score, the more anomalous the item.
 
### Step 4: Fraud Signal Detection
 
A rule-based section:
 
* **Suspicious Like Consistency:** Users whose posts receive very similar like counts (bot-like behavior).
* **Like-to-Play Ratio:** Users with unrealistically high likes compared to video views.
* **Play-to-Like Ratio:** Users with high view counts but low likes (potential view buying).
* **Extreme Like Variation:** Users with large variations in like counts between posts (potential periodic like purchases).
 
### Step 5: Output Generation
 
* **Text Reports (.txt):** Detailed reports for each anomaly and fraud analysis. Also includes a main report and a final summary report.
* **Visualizations (.png):** Four graphic dashboards displaying visual analysis results.
 
---
 
## 5. Installation & Execution Guide
 
### Prerequisites:
 
First, install the required Python libraries:
 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn arabic_reshaper python-bidi openpyxl
```
 
### Configuration (`config.py`):
 
Before running the script, open `config.py` and adjust the following parameters:
 
* `EXCEL_FILE_PATH`: Path to the input Excel file.
* `SHEET_NAME`: Name of the sheet containing data.
* `MIN_SUPPORT`: Minimum support threshold for the ECLAT algorithm (e.g., `0.08` means 8% frequency).
* `IFOREST_CONTAMINATION`: Estimated percentage of anomalous data for Isolation Forest (typically between `0.05` and `0.1`).
* `FONT_NAME_FOR_PERSIAN`: A Persian font installed on your system &#40;like `'Tahoma'` or `'B Nazanin'`&#41; for proper rendering of Persian text in charts.
 
### Execution:
 
To run the full analysis process, simply execute `main.py`:
 
```bash
python main.py
```
 
After execution, a new folder named `results_YYYYMMDD_HHMMSS` will be created containing all outputs.
 
---
 
## 6. Output Files Description
 
Outputs are organized into timestamped folders with the following subdirectories:
 
### `eclat_results/`:
 
* `patterns/`: Text reports of ECLAT patterns by size (e.g., `patterns_size_2.txt`).
* `reports/`:
 
  * `eclat_analysis_report.txt`: Main report summarizing top patterns found across hashtags, users, and time.
 
### `isolation_forest_results/`:
 
* `anomalous_posts_report.txt`: List of posts identified as anomalous with details.
* `anomalous_users_report.txt`: List of anomalous users with their statistical metrics.
* `content_anomalies_report.txt`: List of posts with unusual content characteristics.
* `fraud_signals_report.txt`: Report of users suspected of fraud based on defined rules.
 
### `visualizations/`:
 
* `instagram_analysis_dashboard.png`: Main (2x3) dashboard showing like distributions, like-to-length correlation, posting time patterns, and top Iranian and international users.
* `instagram_overview_dashboard.png`: Overview (2x2) dashboard featuring top users, nationality distribution, performance categories, and most popular hashtags.
* `anomaly_specific_plots.png`: Special anomaly (1x2) plots comparing normal vs. anomalous posts and highlighting users with most anomalous posts.
* `eclat_patterns_visualization.png`: Visualization of top hashtag ECLAT patterns by size.
 
### Root-Level File:
 
* `summary_report_[timestamp].txt`: A comprehensive summary report of all results, key statistics, top findings, and links to all other output files. For a quick overview, start with this file.
 
---
 
## 7. Dependencies and Technologies
 
* **Python 3.x**
* **Pandas:** For data manipulation and analysis
 