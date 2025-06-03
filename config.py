# # config.py

# # --- Input Data Configuration ---
# EXCEL_FILE_PATH = 'Untitled spreadsheet.xlsx'  # Path to your Excel file
# SHEET_NAME = 'videos'                         # Name of the sheet to analyze

# # --- ECLAT Algorithm Parameters ---
# MIN_SUPPORT = 0.04                          # Minimum support for frequent itemsets
# MAX_ECLAT_ITEMSET_SIZE = 7                    # Maximum number of items in an Eclat itemset

# # --- Output Directory Configuration ---
# BASE_OUTPUT_DIR_NAME = "results" # Base name for the main results folder, timestamp will be appended
# ECLAT_RESULTS_DIR = "eclat_results"
# ECLAT_PATTERNS_SUBDIR = "patterns" # Parent for specific Eclat pattern types
# ECLAT_REPORTS_SUBDIR = "reports"
# ISOLATION_FOREST_DIR = "isolation_forest_results"
# VISUALIZATIONS_DIR = "visualizations" # Existing, will be used by all plots

# # --- Subdirectories for Eclat Pattern Types ---
# ECLAT_HASHTAG_PATTERNS_DIR_NAME = "hashtags"
# ECLAT_USER_PATTERNS_DIR_NAME = "users"
# ECLAT_TEMPORAL_PATTERNS_DIR_NAME = "temporal"


# # --- Report File Names ---
# ANOMALOUS_POSTS_REPORT_FILE = "anomalous_posts_report.txt"
# ANOMALOUS_USERS_REPORT_FILE = "anomalous_users_report.txt"
# CONTENT_ANOMALIES_REPORT_FILE = "content_anomalies_report.txt"
# FRAUD_SIGNALS_REPORT_FILE = "fraud_signals_report.txt"
# MAIN_ECLAT_REPORT_FILE = "eclat_analysis_report.txt" 
# VISUALIZATIONS_FILE = "instagram_analysis_dashboard.png" # Main 2x3 dashboard
# SUMMARY_REPORT_FILE = "summary_report.txt" 

# # --- Visualization Configuration (New additions based on user's sample) ---
# TOP_N_DISPLAY = 10  # For top N items in new plots
# FIGURE_SIZE_OVERVIEW = (18, 12) # For the new 2x2 dashboard
# FIGURE_SIZE_ANOMALY_SPECIFIC = (16, 6) # For the anomaly-specific 1x2 plot
# FIGURE_SIZE_PATTERNS_SPECIFIC = (18, 14) # For the Eclat patterns 2x2 plot
# DPI_SETTING = 300

# # --- New Plot Filenames ---
# OVERVIEW_DASHBOARD_FILE = "instagram_overview_dashboard.png"
# ANOMALY_SPECIFIC_VIS_FILE = "anomaly_specific_plots.png"
# ECLAT_PATTERNS_VIS_FILE = "eclat_patterns_visualization.png"

# # Note: Specific Eclat pattern file names like "patterns_size_X.txt"
# # will be generated dynamically in the main_analyzer.py script.



# config.py
import numpy as np # Imported for np.inf

# --- Input Data Configuration ---
EXCEL_FILE_PATH = 'instagramData.xlsx'  # Path to your Excel file
SHEET_NAME = 'videos'                         # Name of the sheet to analyze

# --- Column Names from Excel (Case-sensitive) ---
# These should match the exact column names in your Excel sheet
COL_LIKES_COUNT = 'likesCount'
COL_COMMENTS_COUNT = 'commentsCount'
COL_VIDEO_PLAY_COUNT = 'videoPlayCount'
COL_VIDEO_VIEW_COUNT = 'videoViewCount' # Often similar to play count, but can be different
COL_VIDEO_DURATION = 'videoDuration'
COL_CAPTION = 'caption'
COL_HASHTAGS = 'hashtags'
COL_OWNER_USERNAME = 'ownerUsername'
COL_TIMESTAMP = 'timestamp' # For posting time
COL_POST_ID = 'id' # Unique identifier for each post, if available
COL_POST_TYPE = 'type' # e.g., Video, Image, Sidecar - if available
COL_VIDEO_URL = 'videoUrl' # If available, for reports

# --- Derived Column Names (used internally, less likely to change but can be configured if needed) ---
DERIVED_COL_NATIONALITY = 'nationality'
DERIVED_COL_HOUR = 'hour'
DERIVED_COL_DAY_OF_WEEK = 'day_of_week'
DERIVED_COL_MONTH = 'month'
DERIVED_COL_LIKE_CATEGORY = 'like_category'
DERIVED_COL_IS_ANOMALY = 'is_anomaly' # General anomaly flag for posts/users
DERIVED_COL_ANOMALY_SCORE = 'anomaly_score'

# --- ECLAT Algorithm Parameters ---
MIN_SUPPORT = 0.08                            # Minimum support for frequent itemsets
MAX_ECLAT_ITEMSET_SIZE = 7                    # Maximum number of items in an Eclat itemset

# --- Output Directory Configuration ---
BASE_OUTPUT_DIR_NAME = "results" 
ECLAT_RESULTS_DIR = "eclat_results"
ECLAT_PATTERNS_SUBDIR = "patterns" 
ECLAT_REPORTS_SUBDIR = "reports"
ISOLATION_FOREST_DIR = "isolation_forest_results"
VISUALIZATIONS_DIR = "visualizations" 

# --- Subdirectories for Eclat Pattern Types ---
ECLAT_HASHTAG_PATTERNS_DIR_NAME = "hashtags"
ECLAT_USER_PATTERNS_DIR_NAME = "users"
ECLAT_TEMPORAL_PATTERNS_DIR_NAME = "temporal"

# --- Report File Names ---
ANOMALOUS_POSTS_REPORT_FILE = "anomalous_posts_report.txt"
ANOMALOUS_USERS_REPORT_FILE = "anomalous_users_report.txt"
CONTENT_ANOMALIES_REPORT_FILE = "content_anomalies_report.txt"
FRAUD_SIGNALS_REPORT_FILE = "fraud_signals_report.txt"
MAIN_ECLAT_REPORT_FILE = "eclat_analysis_report.txt" 
SUMMARY_REPORT_FILE = "summary_report.txt" 

# --- Visualization Configuration ---
TOP_N_DISPLAY = 10  
FIGURE_SIZE_MAIN_DASHBOARD = (22, 14) # For the original 2x3 dashboard
FIGURE_SIZE_OVERVIEW = (18, 12) 
FIGURE_SIZE_ANOMALY_SPECIFIC = (16, 6) 
FIGURE_SIZE_PATTERNS_SPECIFIC = (18, 14) 
DPI_SETTING = 300

# --- Plot Filenames ---
VISUALIZATIONS_FILE = "instagram_analysis_dashboard.png" # Main 2x3 dashboard
OVERVIEW_DASHBOARD_FILE = "instagram_overview_dashboard.png"
ANOMALY_SPECIFIC_VIS_FILE = "anomaly_specific_plots.png"
ECLAT_PATTERNS_VIS_FILE = "eclat_patterns_visualization.png"

# --- Like Category Definitions (for preprocess_data and visualizations) ---
LIKE_CATEGORY_BINS = [-np.inf, 100, 1000, 5000, 10000, np.inf]
LIKE_CATEGORY_LABELS = ['Very Low', 'Low', 'Medium', 'High', 'Viral']

# --- Anomaly Detection Parameters ---
IFOREST_CONTAMINATION_POSTS = 0.1
IFOREST_CONTAMINATION_USERS = 0.05
IFOREST_CONTAMINATION_CONTENT = 0.1
RANDOM_STATE_SEED = 42 # For reproducibility

# --- Thresholds for Specific Anomaly/Fraud Rules (Examples) ---
ANOMALOUS_POST_HIGH_LIKE_THRESHOLD = 100000
# POPULAR_HASHTAGS_FOR_COMPARISON = {'#قهوه', '#اسپرسو'} # Removed, will be dynamic
USER_ANOMALY_LOW_ACTIVITY_THRESHOLD = 5 
CONTENT_ANOMALY_SHORT_VIDEO_DURATION_SECONDS = 5
CONTENT_ANOMALY_NO_HASHTAG_MIN_LIKES = 100000

FRAUD_CONSISTENT_LIKES_MIN_POSTS = 5
FRAUD_CONSISTENT_LIKES_MIN_AVG_LIKES = 100
FRAUD_LIKE_TO_PLAY_RATIO_THRESHOLD = 2 
FRAUD_LIKE_MIN_AVG_LIKES = 100
FRAUD_PLAY_TO_LIKE_RATIO_THRESHOLD = 20
FRAUD_PLAY_MIN_AVG_PLAYS = 1000
FRAUD_COMMENT_TO_PLAY_RATIO_THRESHOLD = 0.5
FRAUD_COMMENT_MIN_AVG_COMMENTS = 20





if __name__ == "__main__":
    print("\n\n This file is for config, You Can run 'main.py'\n\n")