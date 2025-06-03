# main_analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import json
from collections import Counter
import re
from itertools import combinations
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
import os

# کتابخانه های لازم برای نمایش صحیح فارسی
import arabic_reshaper
from bidi.algorithm import get_display

# Import from config file
import config

warnings.filterwarnings('ignore')

# --- Font settings for correct Persian display in charts ---
FONT_NAME_FOR_PERSIAN = config.FONT_NAME_FOR_PERSIAN 
try:
    plt.rcParams['font.family'] = FONT_NAME_FOR_PERSIAN
    plt.rcParams['axes.unicode_minus'] = False 
    print(f"✅ Matplotlib font for Persian display in charts set to '{FONT_NAME_FOR_PERSIAN}'.")
    font_path = fm.findfont(fm.FontProperties(family=FONT_NAME_FOR_PERSIAN))
    if not font_path:
        print(f"⚠️ Warning: Font '{FONT_NAME_FOR_PERSIAN}' not found on the system. Persian text in charts might not display correctly.")
        print("Please install a suitable Persian font and specify its name in FONT_NAME_FOR_PERSIAN in the code.")
    else:
        print(f" Font path for '{FONT_NAME_FOR_PERSIAN}': {font_path}")

except Exception as e:
    print(f"❌ Error setting Persian font: {e}. Please install a suitable Persian font and specify its name in the code.")
    print("Continuing with Matplotlib's default font...")
# --- End of font settings ---


class InstagramECLATAnalyzer:
    def __init__(self, excel_file, sheet_name, min_support, max_eclat_itemset_size):
        """Initialize analyzer with parameters from config"""
        self.excel_file = excel_file
        self.sheet_name = sheet_name
        self.min_support = min_support
        self.max_eclat_itemset_size = max_eclat_itemset_size
        self.df = None
        self.processed_df = None
        self.timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Output directory paths
        self.base_output_dir = f"{config.BASE_OUTPUT_DIR_NAME}_{self.timestamp_str}"
        self.eclat_dir = os.path.join(self.base_output_dir, config.ECLAT_RESULTS_DIR)
        self.eclat_patterns_parent_dir = os.path.join(self.eclat_dir, config.ECLAT_PATTERNS_SUBDIR)
        self.eclat_reports_dir = os.path.join(self.eclat_dir, config.ECLAT_REPORTS_SUBDIR)
        self.isolation_forest_dir = os.path.join(self.base_output_dir, config.ISOLATION_FOREST_DIR)
        self.visualizations_dir = os.path.join(self.base_output_dir, config.VISUALIZATIONS_DIR)

        # To store Eclat results for pattern visualization
        self.hashtag_patterns_df = pd.DataFrame()
        self.user_patterns_df = pd.DataFrame()
        self.temporal_patterns_df = pd.DataFrame()

    def _get_persian_display_text(self, text):
        """Helper function to reshape and reorder Persian text for display in charts."""
        if isinstance(text, (int, float, np.integer, np.floating, bool)):
            return str(text)
        if not isinstance(text, str) or not text.strip():
            return text

        if not re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', str(text)):
             return text 
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except Exception:
            return text 

    def _initialize_output_directories(self):
        """Creates the necessary output directories."""
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.eclat_patterns_parent_dir, exist_ok=True)
        os.makedirs(self.eclat_reports_dir, exist_ok=True)
        os.makedirs(self.isolation_forest_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)

        for pattern_type_dir in [config.ECLAT_HASHTAG_PATTERNS_DIR_NAME,
                                 config.ECLAT_USER_PATTERNS_DIR_NAME,
                                 config.ECLAT_TEMPORAL_PATTERNS_DIR_NAME]:
            os.makedirs(os.path.join(self.eclat_patterns_parent_dir, pattern_type_dir), exist_ok=True)

        print(f"\n📁 Output directories created under: {self.base_output_dir}")

    def load_data(self):
        print(f"\n📁 Loading file: {self.excel_file}")
        print("="*60)
        try:
            xl_file = pd.ExcelFile(self.excel_file)
            if self.sheet_name not in xl_file.sheet_names:
                print(f"❌ Error: Sheet '{self.sheet_name}' not found in '{self.excel_file}'. Available: {', '.join(xl_file.sheet_names)}")
                return None
            self.df = pd.read_excel(self.excel_file, sheet_name=self.sheet_name)
            print(f"\n✅ Sheet '{self.sheet_name}' loaded: {len(self.df)} rows, {len(self.df.columns)} columns.")
        except FileNotFoundError:
            print(f"❌ Error: Excel file not found at '{self.excel_file}'.")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        return self.df

    def is_persian_content(self, text):
        if pd.isna(text): return False
        return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', str(text)))

    def detect_user_nationality(self, row):
        caption_text = str(row.get('caption', ''))
        hashtags_text = str(row.get('hashtags', ''))
        username_text = str(row.get('ownerUsername', ''))

        if self.is_persian_content(caption_text): return 'Iranian'
        if self.is_persian_content(hashtags_text): return 'Iranian'
        
        iranian_keywords = config.iranian_keywords
        
        text_to_check = [hashtags_text.lower(), caption_text.lower(), username_text.lower()]
        full_text_lower = " ".join(text_to_check)
        for keyword in iranian_keywords:
            if keyword in full_text_lower: return 'Iranian'
        return 'International'

    def preprocess_data(self):
        if self.df is None: 
            print("❌ DataFrame not loaded. Cannot preprocess.")
            return None
        print("\n🔄 Starting data preprocessing...")
        self.processed_df = self.df.copy()
        
        numeric_cols = ['likesCount', 'commentsCount', 'videoPlayCount', 'videoViewCount', 'videoDuration']
        for col in numeric_cols:
            if col in self.processed_df.columns:
                self.processed_df[col] = pd.to_numeric(self.processed_df[col], errors='coerce').fillna(0)
            else:
                print(f"⚠️ Warning: Numeric column '{col}' not found in DataFrame. Skipping conversion.")
        
        if 'likesCount' in self.processed_df.columns:
            original_rows = len(self.processed_df)
            self.processed_df = self.processed_df[self.processed_df['likesCount'] > 0].copy()
            print(f"📊 Posts filtered (likes > 0): {original_rows} -> {len(self.processed_df)} rows")
        else: 
            print("⚠️ 'likesCount' column missing. Skipping filtering based on positive likes.")

        required_nationality_cols = ['ownerUsername', 'caption', 'hashtags']
        if all(col in self.processed_df.columns for col in required_nationality_cols):
            print("🌍 Detecting user nationalities...")
            self.processed_df['nationality'] = self.processed_df.apply(self.detect_user_nationality, axis=1)
            if 'nationality' in self.processed_df.columns:
                iranian_count = (self.processed_df['nationality'] == 'Iranian').sum()
                international_count = (self.processed_df['nationality'] == 'International').sum()
                print(f"✅ Nationalities: Iranian: {iranian_count}, International: {international_count}")
        else:
            missing_nat_cols = [col for col in required_nationality_cols if col not in self.processed_df.columns]
            print(f"⚠️ Missing columns for nationality detection: {', '.join(missing_nat_cols)}. Assigning 'Unknown'.")
            self.processed_df['nationality'] = 'Unknown'

        if 'timestamp' in self.processed_df.columns:
            self.processed_df['timestamp'] = pd.to_datetime(self.processed_df['timestamp'], errors='coerce')
            self.processed_df['hour'] = self.processed_df['timestamp'].dt.hour
            self.processed_df['day_of_week'] = self.processed_df['timestamp'].dt.day_name()
            print("✅ Time features extracted.")
        else: 
            print("⚠️ 'timestamp' column missing. Skipping time features.")
            self.processed_df['hour'] = -1 
            self.processed_df['day_of_week'] = 'Unknown'
        
        if 'likesCount' in self.processed_df.columns:
            self.processed_df['like_category'] = pd.cut(
                self.processed_df['likesCount'],
                bins=config.LIKE_CATEGORY_BINS, 
                labels=config.LIKE_CATEGORY_LABELS, right=False,
            )
            print("✅ Like categories created.")
        else: 
            print("⚠️ 'likesCount' column missing. Skipping like categories.")
            self.processed_df['like_category'] = 'Unknown'

        print("\n✅ Preprocessing completed!")
        return self.processed_df

    def extract_hashtags(self, hashtag_str):
        if pd.isna(hashtag_str) or hashtag_str == '[]' or hashtag_str == '': return []
        try:
            current_hashtag_str = str(hashtag_str)
            if current_hashtag_str.startswith('['):
                current_hashtag_str = current_hashtag_str.replace("'", '"').replace("None", "null")
                hashtags = json.loads(current_hashtag_str)
            elif isinstance(hashtag_str, list):
                hashtags = hashtag_str
            else: 
                hashtags = re.findall(r'#\w+', current_hashtag_str)
            return ['#' + tag.strip('# ').lower() for tag in hashtags if isinstance(tag, str) and tag.strip('# ')]
        except json.JSONDecodeError:
            return ['#' + tag.strip('# ').lower() for tag in re.findall(r'#\w+', str(hashtag_str)) if tag.strip('# ')]
        except Exception: 
            return []

    def run_eclat(self, transactions, analysis_type_name="generic"):
        if not transactions: return pd.DataFrame()
        all_items = [item for sublist in transactions for item in sublist]
        item_counts = Counter(all_items)
        total_transactions = len(transactions)
        min_count = max(1, int(self.min_support * total_transactions))
        frequent_items_dict = {item: count for item, count in item_counts.items() if count >= min_count}
        if not frequent_items_dict: return pd.DataFrame()

        frequent_items_list = list(frequent_items_dict.keys())
        results = []
        for k in range(1, self.max_eclat_itemset_size + 1):
            if len(frequent_items_list) < k: break
            if k == 1:
                for item in frequent_items_list:
                    results.append({'itemset': item, 'support': frequent_items_dict[item]/total_transactions, 'count': frequent_items_dict[item], 'size': k})
            else:
                for itemset_tuple in combinations(frequent_items_list, k):
                    itemset_candidate = set(itemset_tuple)
                    support_count = sum(1 for trans_set in transactions if itemset_candidate.issubset(set(trans_set)))
                    if support_count >= min_count:
                        results.append({'itemset': ' + '.join(sorted(list(itemset_candidate))), 'support': support_count/total_transactions, 'count': support_count, 'size': k})
        return pd.DataFrame(results).sort_values(['size', 'support'], ascending=[True, False]) if results else pd.DataFrame()

    def _save_eclat_patterns_by_size(self, eclat_df, pattern_type_name):
        if eclat_df is None or eclat_df.empty: return
        specific_patterns_dir = os.path.join(self.eclat_patterns_parent_dir, pattern_type_name)
        os.makedirs(specific_patterns_dir, exist_ok=True)
        for size, group_df in eclat_df.groupby('size'):
            file_path = os.path.join(specific_patterns_dir, f"patterns_size_{size}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"ECLAT Patterns for {pattern_type_name} - Itemset Size: {size}\n" + "="*50 + "\n")
                for _, row in group_df.sort_values('support', ascending=False).iterrows():
                    f.write(f"Itemset: {row['itemset']}\n  Support: {row['support']:.4f} ({row['count']} occurrences)\n\n")
        print(f"💾 All {pattern_type_name} Eclat patterns saved to: {specific_patterns_dir}")

    def analyze_hashtag_patterns(self):
        print("\n🔍 Analyzing hashtag combinations...")
        if not (self.processed_df is not None and all(c in self.processed_df.columns for c in ['hashtags', 'likesCount'])):
            print("❌ Missing 'hashtags' or 'likesCount'. Skipping hashtag pattern analysis.")
            return pd.DataFrame()
        transactions = [list(set(self.extract_hashtags(row.get('hashtags','')) + ['viral_performance' if row.get('likesCount',0)>5000 else 'normal_performance'])) for _, row in self.processed_df.iterrows() if self.extract_hashtags(row.get('hashtags',''))]
        if not transactions: return pd.DataFrame()
        self.hashtag_patterns_df = self.run_eclat(transactions, config.ECLAT_HASHTAG_PATTERNS_DIR_NAME)
        if not self.hashtag_patterns_df.empty: self._save_eclat_patterns_by_size(self.hashtag_patterns_df, config.ECLAT_HASHTAG_PATTERNS_DIR_NAME)
        return self.hashtag_patterns_df
    
    def analyze_user_patterns(self):
        print("\n👥 Analyzing user patterns...")
        if not (self.processed_df is not None and all(c in self.processed_df.columns for c in ['ownerUsername', 'likesCount'])):
            print("❌ Missing 'ownerUsername' or 'likesCount'. Skipping user pattern analysis.")
            return pd.DataFrame()
        transactions = []
        for username in self.processed_df['ownerUsername'].unique():
            user_data = self.processed_df[self.processed_df['ownerUsername'] == username]
            if user_data.empty: continue
            items = []
            post_count = len(user_data)
            if post_count >= 50: items.append('high_activity')
            elif post_count >= 20: items.append('medium_activity')
            else: items.append('low_activity')
            
            avg_likes = user_data['likesCount'].mean() if 'likesCount' in user_data and not user_data['likesCount'].empty else 0
            if avg_likes >= 5000: items.append('high_engagement')
            elif avg_likes >= 1000: items.append('medium_engagement')
            else: items.append('low_engagement')
            
            if 'hour' in user_data.columns: 
                most_common_hour = user_data['hour'].mode()
                if not most_common_hour.empty:
                    hour = most_common_hour.iloc[0]
                    if 6 <= hour < 12: items.append('morning_poster')
                    elif 12 <= hour < 18: items.append('afternoon_poster')
                    elif 18 <= hour < 24: items.append('evening_poster')
                    else: items.append('night_poster')
            if items: transactions.append(list(set(items)))
        if not transactions: return pd.DataFrame()
        self.user_patterns_df = self.run_eclat(transactions, config.ECLAT_USER_PATTERNS_DIR_NAME)
        if not self.user_patterns_df.empty: self._save_eclat_patterns_by_size(self.user_patterns_df, config.ECLAT_USER_PATTERNS_DIR_NAME)
        return self.user_patterns_df
    
    def analyze_temporal_patterns(self):
        print("\n⏰ Analyzing temporal patterns...")
        if not (self.processed_df is not None and 'timestamp' in self.processed_df.columns):
            print("❌ Missing 'timestamp'. Skipping temporal pattern analysis.")
            return pd.DataFrame()
        valid_df = self.processed_df[self.processed_df['timestamp'].notna()].copy()
        if valid_df.empty: return pd.DataFrame()
        transactions = []
        for _, row in valid_df.iterrows():
            items = []
            if 'day_of_week' in row and pd.notna(row['day_of_week']): items.append(f"day_{row['day_of_week']}")
            if 'hour' in row and pd.notna(row['hour']):
                hour = row['hour']
                if 6 <= hour < 12: items.append('morning_post')
                elif 12 <= hour < 18: items.append('afternoon_post')
                elif 18 <= hour < 24: items.append('evening_post')
                else: items.append('night_post')
            if 'likesCount' in row and row.get('likesCount', 0) > 5000: items.append('high_performance_post')
            elif 'likesCount' in row : items.append('normal_performance_post') 
            if 'type' in row and pd.notna(row['type']): items.append(f"type_{row['type']}") 
            if items: transactions.append(list(set(items)))
        if not transactions: return pd.DataFrame()
        self.temporal_patterns_df = self.run_eclat(transactions, config.ECLAT_TEMPORAL_PATTERNS_DIR_NAME)
        if not self.temporal_patterns_df.empty: self._save_eclat_patterns_by_size(self.temporal_patterns_df, config.ECLAT_TEMPORAL_PATTERNS_DIR_NAME)
        return self.temporal_patterns_df

    def _format_category_details(self, category_df, id_col, main_metric_col, other_metrics_cols, score_col_name='anomaly_score'):
        """
        Formats the detailed list of anomalies for a specific category.
        """
        report_lines = []
        if category_df.empty:
            # Adding a space before the message to align with numbered items if they were present.
            report_lines.append("    No anomalies found in this category.") 
            return "\n".join(report_lines)

        # Ensure the DataFrame is sorted by the score column
        # Handle cases where score_col_name might not exist (e.g., if a df is passed without it)
        if score_col_name in category_df.columns:
            sorted_category_df = category_df.sort_values(score_col_name, ascending=True)
        else:
            print(f"⚠️ Warning: Score column '{score_col_name}' not found in DataFrame for formatting. Using original order.")
            sorted_category_df = category_df


        for idx, (item_id_or_index, row) in enumerate(sorted_category_df.iterrows(), 1):
            identifier_str = f"@{row[id_col]}" if id_col and id_col in row and pd.notna(row[id_col]) else f"Item Index: {item_id_or_index}"
            if id_col is None: # For user anomalies where index is the username
                identifier_str = f"@{item_id_or_index}"

            report_lines.append(f"\n{idx}. {identifier_str}") # Added newline before each item for spacing
            if main_metric_col in row and pd.notna(row[main_metric_col]):
                report_lines.append(f"    Main Metric ({main_metric_col.replace('_', ' ').title()}): {row[main_metric_col]:,.2f}")

            details = []
            for col_name in other_metrics_cols:
                # Ensure the column exists and is not the score column itself, and has a non-null value
                if col_name in row and pd.notna(row[col_name]) and col_name != score_col_name:
                    value = row[col_name]
                    col_name_english = col_name.replace('_', ' ').title()
                    
                    # Truncate long video URLs for display
                    if col_name == config.COL_VIDEO_URL and isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    elif isinstance(value, list) and col_name == config.COL_VIDEO_URL: # Handle list of dicts for videoUrl
                        try:
                            urls = [item.get('url', '') for item in value if isinstance(item, dict)]
                            urls_str = "; ".join(urls)
                            if len(urls_str) > 100: urls_str = urls_str[:100] + "..."
                            value = urls_str if urls_str else "Not available"
                        except: value = "Error parsing video URL list"


                    if isinstance(value, float):
                        details.append(f"{col_name_english}: {value:.2f}")
                    else:
                        details.append(f"{col_name_english}: {str(value)}") # Ensure value is string
            
            # Add anomaly score to details if it exists and is not null
            current_score_val = row.get(score_col_name)
            if pd.notna(current_score_val):
                 details.append(f"Anomaly Score: {current_score_val:.3f}")
            
            if details:
                report_lines.append(f"    Details: {'; '.join(details)}")
            # If only score was to be shown and no other details, it's already appended if present.
        return "\n".join(report_lines)


    def detect_anomalous_posts(self):
        print("\n🔍 Detecting anomalous posts...")
        report_text_intro = ["="*70 + f"\n🚨 ANOMALOUS POSTS DETECTION REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n" + "="*70]
        specific_findings_summary = {} # This will store DFs for the main summary later

        if self.processed_df is None or self.processed_df.empty:
            return pd.DataFrame(), "\n".join(report_text_intro) + "\nError: Processed data is empty.", specific_findings_summary

        df_for_detection = self.processed_df.copy()

        if 'likesCount' in df_for_detection.columns and 'commentsCount' in df_for_detection.columns:
            df_for_detection['engagement_rate'] = np.where(
                df_for_detection['likesCount'] > 0,
                df_for_detection['commentsCount'] / df_for_detection['likesCount'], 0
            )
        if 'likesCount' in df_for_detection.columns and 'videoPlayCount' in df_for_detection.columns:
            df_for_detection['play_rate'] = np.where(
                df_for_detection['likesCount'] > 0,
                df_for_detection['videoPlayCount'] / df_for_detection['likesCount'], 0
            )
        if 'videoPlayCount' in df_for_detection.columns and 'likesCount' in df_for_detection.columns:
            df_for_detection['likes_per_play'] = np.where(
                df_for_detection['videoPlayCount'] > 0,
                df_for_detection['likesCount'] / df_for_detection['videoPlayCount'], 0
            )
        
        feature_list_for_model = ['likesCount', 'commentsCount', 'videoPlayCount', 'engagement_rate', 'play_rate']
        actual_features_for_model = [f for f in feature_list_for_model if f in df_for_detection.columns]
        
        if not actual_features_for_model or len(actual_features_for_model) < 2:
            error_msg = f"\nError: Not enough valid features for Isolation Forest model. Available after derivation: {', '.join(actual_features_for_model)}."
            for col in ['engagement_rate', 'play_rate', 'likes_per_play']:
                if col in df_for_detection.columns:
                    self.processed_df[col] = df_for_detection[col]
            return pd.DataFrame(), "\n".join(report_text_intro) + error_msg, specific_findings_summary

        X = df_for_detection[actual_features_for_model].copy().fillna(0).replace([np.inf, -np.inf], 0)
        if X.empty:
            for col in ['engagement_rate', 'play_rate', 'likes_per_play']:
                if col in df_for_detection.columns:
                    self.processed_df[col] = df_for_detection[col]
            return pd.DataFrame(), "\n".join(report_text_intro) + "\nError: Feature set X is empty for Isolation Forest.", specific_findings_summary

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        iso_forest = IsolationForest(contamination=config.IFOREST_CONTAMINATION_POSTS, random_state=config.RANDOM_STATE_SEED)
        
        self.processed_df.loc[X.index, 'is_anomaly'] = iso_forest.fit_predict(X_scaled) == -1
        self.processed_df.loc[X.index, 'anomaly_score'] = iso_forest.decision_function(X_scaled)
        
        for col in ['engagement_rate', 'play_rate', 'likes_per_play']:
            if col in df_for_detection.columns:
                self.processed_df.loc[df_for_detection.index, col] = df_for_detection[col]
        
        anomalous_posts_df = self.processed_df[self.processed_df['is_anomaly'] == True].sort_values('anomaly_score', ascending=True)
        
        # --- Summary Section ---
        report_text_intro.append(f"\n📊 Total posts analyzed: {len(self.processed_df)}")
        report_text_intro.append(f"🔴 Anomalies (general) detected: {len(anomalous_posts_df)} ({(len(anomalous_posts_df)/len(self.processed_df)*100 if len(self.processed_df) > 0 else 0.0):.1f}%)")
        
        high_like_threshold = config.ANOMALOUS_POST_HIGH_LIKE_THRESHOLD 
        positive_high_likes = anomalous_posts_df[anomalous_posts_df['likesCount'] > high_like_threshold]
        report_text_intro.append(f"\n🌟 Positive Anomalies: High Likes (> {high_like_threshold:,} LIKES)")
        report_text_intro.append(f"    Found: {len(positive_high_likes)} posts")
        specific_findings_summary['Positive - Very High Likes'] = positive_high_likes[['ownerUsername', 'likesCount', 'anomaly_score']] # For main summary

        if 'hashtags' in self.processed_df.columns and not self.processed_df['hashtags'].empty:
            all_hashtags_list = [tag for sublist in self.processed_df['hashtags'].dropna().apply(self.extract_hashtags) for tag in sublist]
            if all_hashtags_list:
                top_n_hashtags = config.TOP_N_DISPLAY // 2 if config.TOP_N_DISPLAY > 1 else 1
                popular_hashtags_for_comparison_dynamic = set(dict(Counter(all_hashtags_list).most_common(top_n_hashtags)).keys())
                if not popular_hashtags_for_comparison_dynamic: 
                    popular_hashtags_for_comparison_dynamic = {'#coffee', '#espresso'} 
            else:
                popular_hashtags_for_comparison_dynamic = {'#coffee', '#espresso'}
        else:
            popular_hashtags_for_comparison_dynamic = {'#coffee', '#espresso'}

        def has_no_popular_hashtags(row_hashtags_val):
            extracted = self.extract_hashtags(row_hashtags_val)
            return not any(ht in popular_hashtags_for_comparison_dynamic for ht in extracted)

        viral_no_popular_ht = anomalous_posts_df[
            (anomalous_posts_df['like_category'] == 'Viral') & 
            (anomalous_posts_df['hashtags'].apply(has_no_popular_hashtags))
        ]
        report_text_intro.append(f"\n🌟 Positive Anomalies: Viral Without Popular Hashtags (e.g., {', '.join(list(popular_hashtags_for_comparison_dynamic))})")
        report_text_intro.append(f"    Found: {len(viral_no_popular_ht)} posts")
        specific_findings_summary['Positive - Viral without Popular Hashtags'] = viral_no_popular_ht[['ownerUsername', 'likesCount', 'hashtags', 'anomaly_score']]

        negative_zero_likes = anomalous_posts_df[anomalous_posts_df['likesCount'] <= 1] 
        report_text_intro.append(f"\n📉 Negative Anomalies: Zero or Extremely Low Likes (<=1)")
        report_text_intro.append(f"    Found: {len(negative_zero_likes)} posts")
        specific_findings_summary['Negative - Zero/Low Likes'] = negative_zero_likes[['ownerUsername', 'likesCount', 'anomaly_score']]

        low_performance_popular_ht = anomalous_posts_df[
            (anomalous_posts_df['like_category'].isin(['Very Low', 'Low'])) &
            (~anomalous_posts_df['hashtags'].apply(has_no_popular_hashtags)) 
        ]
        report_text_intro.append(f"\n📉 Negative Anomalies: Popular Hashtags but Low Performance")
        report_text_intro.append(f"    Found: {len(low_performance_popular_ht)} posts")
        specific_findings_summary['Negative - Popular Hashtags, Low Performance'] = low_performance_popular_ht[['ownerUsername', 'likesCount', 'hashtags', 'anomaly_score']]
        
        # --- Detailed Categorized Listing ---
        categorized_report_parts = ["\n" + "="*70] # Start detailed section with a separator
        
        # Define categories and their data for iteration
        categories_to_report = [
            (f"🌟 Positive Anomalies: High Likes (> {high_like_threshold:,} LIKES)", positive_high_likes),
            (f"🌟 Positive Anomalies: Viral Without Popular Hashtags (e.g., {', '.join(list(popular_hashtags_for_comparison_dynamic))})", viral_no_popular_ht),
            (f"📉 Negative Anomalies: Zero or Extremely Low Likes (<=1)", negative_zero_likes),
            (f"📉 Negative Anomalies: Popular Hashtags but Low Performance", low_performance_popular_ht)
        ]
        
        other_metrics_cols_posts = ['commentsCount', 'videoPlayCount', 'engagement_rate', 'play_rate', 'nationality', 'videoDuration', config.COL_VIDEO_URL, 'hashtags']

        for i, (title, df_category) in enumerate(categories_to_report):
            categorized_report_parts.append(f"\n{title}") # Add newline before title
            details = self._format_category_details(
                df_category,
                id_col='ownerUsername',
                main_metric_col='likesCount',
                other_metrics_cols=other_metrics_cols_posts,
                score_col_name='anomaly_score'
            )
            categorized_report_parts.append(details)
            if i < len(categories_to_report) - 1: # Add separator if not the last category
                 categorized_report_parts.append("\n" + "-"*70)


        final_report_text = "\n".join(report_text_intro) + "\n" + "\n".join(categorized_report_parts)

        with open(os.path.join(self.isolation_forest_dir, config.ANOMALOUS_POSTS_REPORT_FILE), 'w', encoding='utf-8') as f: f.write(final_report_text)
        print(f"✅ Anomalous posts: {len(anomalous_posts_df)} found. Report saved.")
        return anomalous_posts_df, final_report_text, specific_findings_summary

    def detect_anomalous_users(self):
        print("\n👥 Detecting anomalous users...")
        report_text_intro = ["="*70 + f"\n🚨 ANOMALOUS USERS DETECTION REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n" + "="*70]
        specific_findings_summary = {}

        if not (self.processed_df is not None and not self.processed_df.empty and 'ownerUsername' in self.processed_df.columns):
            return pd.DataFrame(), "\n".join(report_text_intro) + "\nError: Missing data for user anomalies.", specific_findings_summary

        required_agg_cols = ['likesCount', 'commentsCount', 'videoPlayCount', 'like_category'] 
        if not all(col in self.processed_df.columns for col in required_agg_cols):
            missing = [col for col in required_agg_cols if col not in self.processed_df.columns]
            return pd.DataFrame(), "\n".join(report_text_intro) + f"\nError: Missing columns for user aggregation: {', '.join(missing)}.", specific_findings_summary

        user_stats = self.processed_df.groupby('ownerUsername').agg(
            avg_likes=('likesCount', 'mean'), std_likes=('likesCount', 'std'),
            post_count=('likesCount', 'count'), max_likes=('likesCount', 'max'),
            avg_comments=('commentsCount', 'mean'), avg_plays=('videoPlayCount', 'mean'),
        ).round(2).fillna(0)
        if user_stats.empty: return pd.DataFrame(), "\n".join(report_text_intro) + "\nError: User statistics empty.", specific_findings_summary

        user_stats['likes_variation_coeff'] = np.where(user_stats['avg_likes'] > 0, user_stats['std_likes'] / user_stats['avg_likes'], 0)
        user_stats['avg_engagement_rate'] = np.where(user_stats['avg_likes'] > 0, user_stats['avg_comments'] / user_stats['avg_likes'], 0)
        user_stats = user_stats.replace([np.inf, -np.inf], 0)

        feature_list = ['avg_likes', 'post_count', 'avg_comments', 'avg_plays', 'likes_variation_coeff', 'avg_engagement_rate']
        X = user_stats[feature_list].copy().fillna(0)
        if X.empty: return pd.DataFrame(), "\n".join(report_text_intro) + "\nError: Feature set X empty for users.", specific_findings_summary
        
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
        iso_forest = IsolationForest(contamination=config.IFOREST_CONTAMINATION_USERS, random_state=config.RANDOM_STATE_SEED)
        user_stats['is_anomaly'] = iso_forest.fit_predict(X_scaled) == -1
        user_stats['anomaly_score'] = iso_forest.decision_function(X_scaled)
        anomalous_users_df = user_stats[user_stats['is_anomaly']].sort_values('anomaly_score', ascending=True)

        # --- Summary Section ---
        report_text_intro.append(f"\n📊 Total users analyzed: {len(user_stats)}")
        report_text_intro.append(f"🔴 Anomalies (general) detected: {len(anomalous_users_df)} ({(len(anomalous_users_df)/len(user_stats)*100 if len(user_stats) > 0 else 0.0):.1f}%)")


        low_activity_threshold = config.USER_ANOMALY_LOW_ACTIVITY_THRESHOLD 
        high_engagement_threshold_val = user_stats['avg_engagement_rate'].quantile(0.75) if not user_stats['avg_engagement_rate'].empty else 0.1 # Default if quantile fails
        
        low_act_high_eng = anomalous_users_df[
            (anomalous_users_df['post_count'] <= low_activity_threshold) &
            (anomalous_users_df['avg_engagement_rate'] >= high_engagement_threshold_val)
        ]
        report_text_intro.append(f"\n👤 Users with Low Activity & High Engagement (Posts <= {low_activity_threshold}, Engagement Rate >= {high_engagement_threshold_val:.2f})")
        report_text_intro.append(f"    Found: {len(low_act_high_eng)} users")
        specific_findings_summary['Users - Low Activity, High Engagement'] = low_act_high_eng[['avg_likes', 'post_count', 'avg_engagement_rate', 'anomaly_score']]

        users_all_viral_list = [] 
        for username_val in anomalous_users_df.index: 
            user_posts_df = self.processed_df[self.processed_df['ownerUsername'] == username_val] 
            if not user_posts_df.empty and 'like_category' in user_posts_df.columns and all(user_posts_df['like_category'] == 'Viral'):
                users_all_viral_list.append(username_val)
        
        all_viral_df = anomalous_users_df[anomalous_users_df.index.isin(users_all_viral_list)]
        report_text_intro.append(f"\n🤖 Users with 100% Viral Posts (Potential Bots/Inorganic)")
        report_text_intro.append(f"    Found: {len(all_viral_df)} users")
        specific_findings_summary['Users - 100% Viral Posts'] = all_viral_df[['avg_likes', 'post_count', 'anomaly_score']]
        
        if not anomalous_users_df.empty and 'nationality' in self.processed_df.columns:
            nationalities = self.processed_df.groupby('ownerUsername')['nationality'].first()
            anomalous_users_df = anomalous_users_df.join(nationalities, how='left') # Join nationalities to anomalous_users_df
            # Also ensure low_act_high_eng and all_viral_df get nationality if they are used in specific_findings_summary
            if not low_act_high_eng.empty: low_act_high_eng = low_act_high_eng.join(nationalities, how='left')
            if not all_viral_df.empty: all_viral_df = all_viral_df.join(nationalities, how='left')

        elif not anomalous_users_df.empty: 
            anomalous_users_df['nationality'] = 'Unknown'
            if not low_act_high_eng.empty: low_act_high_eng['nationality'] = 'Unknown'
            if not all_viral_df.empty: all_viral_df['nationality'] = 'Unknown'
        
        # --- Detailed Categorized Listing ---
        categorized_report_parts = ["\n" + "="*70]
        other_metrics_cols_users = ['post_count', 'max_likes', 'avg_engagement_rate', 'likes_variation_coeff', 'nationality']
        
        categories_to_report_users = [
            (f"👤 Users with Low Activity & High Engagement (Posts <= {low_activity_threshold}, Engagement Rate >= {high_engagement_threshold_val:.2f})", low_act_high_eng),
            (f"🤖 Users with 100% Viral Posts (Potential Bots/Inorganic)", all_viral_df)
        ]

        for i, (title, df_category) in enumerate(categories_to_report_users):
            categorized_report_parts.append(f"\n{title}")
            details = self._format_category_details(
                df_category,
                id_col=None, # Username is the index
                main_metric_col='avg_likes',
                other_metrics_cols=other_metrics_cols_users,
                score_col_name='anomaly_score'
            )
            categorized_report_parts.append(details)
            if i < len(categories_to_report_users) - 1:
                 categorized_report_parts.append("\n" + "-"*70)
        
        # Add a section for all other general anomalies if any exist beyond specific categories
        general_anomalous_users_not_in_specific = anomalous_users_df[
            ~anomalous_users_df.index.isin(low_act_high_eng.index) &
            ~anomalous_users_df.index.isin(all_viral_df.index)
        ]
        if not general_anomalous_users_not_in_specific.empty:
            categorized_report_parts.append("\n" + "-"*70) # Separator
            categorized_report_parts.append("\n👤 Other General Anomalous Users (by score):")
            details_general = self._format_category_details(
                general_anomalous_users_not_in_specific,
                id_col=None,
                main_metric_col='avg_likes',
                other_metrics_cols=other_metrics_cols_users,
                score_col_name='anomaly_score'
            )
            categorized_report_parts.append(details_general)


        final_report_text = "\n".join(report_text_intro) + "\n" + "\n".join(categorized_report_parts)

        with open(os.path.join(self.isolation_forest_dir, config.ANOMALOUS_USERS_REPORT_FILE), 'w', encoding='utf-8') as f: f.write(final_report_text)
        print(f"✅ Anomalous users: {len(anomalous_users_df)} found. Report saved.")
        return anomalous_users_df, final_report_text, specific_findings_summary

    def detect_content_anomalies(self):
        print("\n📝 Detecting content anomalies (excluding type-based)...")
        report_text_intro = ["="*70 + f"\n🚨 CONTENT ANOMALIES DETECTION REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n" + "="*70]
        specific_findings_summary = {}

        if self.processed_df is None or self.processed_df.empty:
            return pd.DataFrame(), "\n".join(report_text_intro) + "\nError: Processed data is empty.", specific_findings_summary

        content_df = self.processed_df.copy() # Use a copy for this detection
        
        if 'caption' in content_df.columns: content_df['caption_length'] = content_df['caption'].fillna('').astype(str).str.len()
        else: content_df['caption_length'] = 0
        
        if 'hashtags' in content_df.columns: content_df['hashtag_count'] = content_df['hashtags'].fillna('').apply(lambda x: len(self.extract_hashtags(x)))
        else: content_df['hashtag_count'] = 0
        
        # Ensure likes_per_play is on content_df (it might be on self.processed_df from post anomaly detection)
        if 'videoPlayCount' in content_df.columns and 'likesCount' in content_df.columns:
            if 'likes_per_play' not in content_df.columns: # Recalculate if not present on the copy
                 content_df['likes_per_play'] = np.where(
                    content_df['videoPlayCount'] > 0,
                    content_df['likesCount'] / content_df['videoPlayCount'], 0
                )
        elif 'likes_per_play' not in content_df.columns: # If base columns missing and not derived
            content_df['likes_per_play'] = 0


        feature_list_for_model = ['caption_length', 'hashtag_count', 'likesCount', 'commentsCount', 'videoPlayCount', 'videoDuration', 'likes_per_play']
        actual_features_for_model = [f for f in feature_list_for_model if f in content_df.columns and content_df[f].notna().any()] # Ensure column has some non-NA data

        if not actual_features_for_model or len(actual_features_for_model) < 2:
            error_msg = f"\nError: Not enough valid features for Content Anomaly model. Available: {', '.join(actual_features_for_model)}."
            # Copy derived features to self.processed_df before returning, if they were created on content_df
            for col in ['caption_length', 'hashtag_count', 'likes_per_play']:
                if col in content_df.columns: self.processed_df.loc[content_df.index, col] = content_df[col]
            return pd.DataFrame(), "\n".join(report_text_intro) + error_msg, specific_findings_summary

        X = content_df[actual_features_for_model].copy().fillna(0).replace([np.inf, -np.inf], 0)
        if X.empty: 
            for col in ['caption_length', 'hashtag_count', 'likes_per_play']:
                if col in content_df.columns: self.processed_df.loc[content_df.index, col] = content_df[col]
            return pd.DataFrame(), "\n".join(report_text_intro) + "\nError: Feature set X for Content Anomaly is empty.", specific_findings_summary
        
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
        iso_forest = IsolationForest(contamination=config.IFOREST_CONTAMINATION_CONTENT, random_state=config.RANDOM_STATE_SEED)
        
        # Add results to self.processed_df (the main DataFrame)
        self.processed_df.loc[X.index, 'is_content_anomaly'] = iso_forest.fit_predict(X_scaled) == -1
        self.processed_df.loc[X.index, 'content_anomaly_score'] = iso_forest.decision_function(X_scaled)

        # Update derived features on self.processed_df from content_df (where they were calculated)
        for col in ['caption_length', 'hashtag_count', 'likes_per_play']:
            if col in content_df.columns:
                self.processed_df.loc[content_df.index, col] = content_df[col]
        
        anomalous_content_df = self.processed_df[self.processed_df['is_content_anomaly'] == True].sort_values('content_anomaly_score', ascending=True)
        
        # --- Summary Section ---
        report_text_intro.append(f"\n📊 Total posts analyzed for content: {len(self.processed_df)}")
        report_text_intro.append(f"🔴 Content Anomalies (general) detected: {len(anomalous_content_df)} ({(len(anomalous_content_df)/len(self.processed_df)*100 if len(self.processed_df) > 0 else 0.0):.1f}%)")

        short_viral_videos = pd.DataFrame()
        if 'videoDuration' in anomalous_content_df.columns and 'like_category' in anomalous_content_df.columns:
            short_viral_videos = anomalous_content_df[
                (anomalous_content_df['videoDuration'] < config.CONTENT_ANOMALY_SHORT_VIDEO_DURATION_SECONDS) & (anomalous_content_df['videoDuration'] > 0) &
                (anomalous_content_df['like_category'] == 'Viral')
            ]
        report_text_intro.append(f"\n📱 Unusual Content: Viral Videos Shorter Than {config.CONTENT_ANOMALY_SHORT_VIDEO_DURATION_SECONDS} Seconds")
        report_text_intro.append(f"    Found: {len(short_viral_videos)} posts")
        specific_findings_summary['Content - Short Viral Videos (<5s)'] = short_viral_videos[['ownerUsername', 'likesCount', 'videoDuration', 'content_anomaly_score']]


        no_hashtag_high_likes = pd.DataFrame()
        if 'hashtag_count' in anomalous_content_df.columns and 'likesCount' in anomalous_content_df.columns: # hashtag_count is from content_df
            no_hashtag_high_likes = anomalous_content_df[
                (anomalous_content_df['hashtag_count'] == 0) & # Use hashtag_count from anomalous_content_df
                (anomalous_content_df['likesCount'] > config.CONTENT_ANOMALY_NO_HASHTAG_MIN_LIKES)
            ]
        report_text_intro.append(f"\n📱 Unusual Content: No Hashtags but >{config.CONTENT_ANOMALY_NO_HASHTAG_MIN_LIKES:,} Likes")
        report_text_intro.append(f"    Found: {len(no_hashtag_high_likes)} posts")
        specific_findings_summary['Content - No Hashtags, High Likes'] = no_hashtag_high_likes[['ownerUsername', 'likesCount', 'content_anomaly_score']]
        
        # --- Detailed Categorized Listing ---
        categorized_report_parts = ["\n" + "="*70]
        other_metrics_cols_content = ['videoPlayCount', 'caption_length', 'hashtag_count', 'likes_per_play', 'nationality', config.COL_VIDEO_URL, 'videoDuration']
        
        categories_to_report_content = [
            (f"📱 Unusual Content: Viral Videos Shorter Than {config.CONTENT_ANOMALY_SHORT_VIDEO_DURATION_SECONDS} Seconds", short_viral_videos),
            (f"📱 Unusual Content: No Hashtags but >{config.CONTENT_ANOMALY_NO_HASHTAG_MIN_LIKES:,} Likes", no_hashtag_high_likes)
        ]

        for i, (title, df_category) in enumerate(categories_to_report_content):
            categorized_report_parts.append(f"\n{title}")
            details = self._format_category_details(
                df_category,
                id_col='ownerUsername',
                main_metric_col='likesCount',
                other_metrics_cols=other_metrics_cols_content,
                score_col_name='content_anomaly_score' # Specific score for content
            )
            categorized_report_parts.append(details)
            if i < len(categories_to_report_content) - 1:
                 categorized_report_parts.append("\n" + "-"*70)

        # Add a section for all other general content anomalies
        general_content_anomalies_not_in_specific = anomalous_content_df[
            ~anomalous_content_df.index.isin(short_viral_videos.index) &
            ~anomalous_content_df.index.isin(no_hashtag_high_likes.index)
        ]
        if not general_content_anomalies_not_in_specific.empty:
            categorized_report_parts.append("\n" + "-"*70) 
            categorized_report_parts.append("\n📄 Other General Content Anomalies (by score):")
            details_general_content = self._format_category_details(
                general_content_anomalies_not_in_specific,
                id_col='ownerUsername',
                main_metric_col='likesCount',
                other_metrics_cols=other_metrics_cols_content,
                score_col_name='content_anomaly_score'
            )
            categorized_report_parts.append(details_general_content)


        final_report_text = "\n".join(report_text_intro) + "\n" + "\n".join(categorized_report_parts)

        with open(os.path.join(self.isolation_forest_dir, config.CONTENT_ANOMALIES_REPORT_FILE), 'w', encoding='utf-8') as f: f.write(final_report_text)
        print(f"✅ Content anomalies: {len(anomalous_content_df)} found. Report saved.")
        return anomalous_content_df, final_report_text, specific_findings_summary

    def detect_fraud_signals(self):
        print("\n🚨 Detecting fraud signals...")
        
        if not (self.processed_df is not None and not self.processed_df.empty and 
                'ownerUsername' in self.processed_df.columns and
                'likesCount' in self.processed_df.columns and
                'commentsCount' in self.processed_df.columns and
                'videoPlayCount' in self.processed_df.columns):
            error_msg = "Error: Missing required data/columns (ownerUsername, likesCount, commentsCount, videoPlayCount) for fraud signal detection."
            print(f"❌ {error_msg}")
            return pd.DataFrame(), error_msg, {}

        agg_dict = {
            'likesCount': ['mean', 'std', 'min', 'max'],
            'commentsCount': ['mean', 'std'],
            'videoPlayCount': ['mean', 'std'],
        }
        if config.COL_POST_ID in self.processed_df.columns:
            agg_dict[config.COL_POST_ID] = 'count'
            user_metrics = self.processed_df.groupby('ownerUsername').agg(agg_dict).round(2)
            user_metrics.columns = ['avg_likes', 'std_likes', 'min_likes', 'max_likes',
                                    'avg_comments', 'std_comments', 'avg_plays', 'std_plays', 'post_count']
        else:
            print(f"⚠️ Column '{config.COL_POST_ID}' not found for post_count. Counting posts per user by row count.")
            user_metrics = self.processed_df.groupby('ownerUsername').agg(agg_dict).round(2)
            if isinstance(user_metrics.columns, pd.MultiIndex):
                user_metrics.columns = ["_".join(col).strip() for col in user_metrics.columns.values]
                rename_map = {
                    'likesCount_mean': 'avg_likes', 'likesCount_std': 'std_likes', 
                    'likesCount_min': 'min_likes', 'likesCount_max': 'max_likes',
                    'commentsCount_mean': 'avg_comments', 'commentsCount_std': 'std_comments',
                    'videoPlayCount_mean': 'avg_plays', 'videoPlayCount_std': 'std_plays'
                }
                user_metrics = user_metrics.rename(columns=rename_map)
            user_metrics['post_count'] = self.processed_df.groupby('ownerUsername').size()
        
        user_metrics = user_metrics.fillna(0)

        if user_metrics.empty:
            error_msg = "Error: User metrics are empty after aggregation."
            print(f"❌ {error_msg}")
            return pd.DataFrame(), error_msg, {}
        
        user_metrics['like_spread'] = user_metrics['max_likes'] - user_metrics['min_likes']
        user_metrics['consistency_score'] = np.where(
            user_metrics['avg_likes'] > 0,
            user_metrics['std_likes'] / (user_metrics['avg_likes'] + 1e-6), 
            0
        )
        
        user_metrics['like_to_play_ratio'] = np.where(
            user_metrics['avg_plays'] > 0,
            user_metrics['avg_likes'] / (user_metrics['avg_plays'] + 1e-6),
            0 
        )
        user_metrics['play_to_like_ratio'] = np.where(
            user_metrics['avg_likes'] > 0,
            user_metrics['avg_plays'] / (user_metrics['avg_likes'] + 1e-6),
            0
        )
        user_metrics['comment_to_play_ratio'] = np.where(
            user_metrics['avg_plays'] > 0,
            user_metrics['avg_comments'] / (user_metrics['avg_plays'] + 1e-6),
            0
        )
        
        user_metrics = user_metrics.replace([np.inf, -np.inf], 0).fillna(0)
        
        report_lines = []
        report_lines.append("="*70) 
        report_lines.append(f"🚨 FRAUD SIGNALS DETECTION REPORT ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
        report_lines.append("="*70)
        report_lines.append("")
        report_lines.append("⚠️ SUSPICIOUS PATTERNS DETECTED:")
        report_lines.append("")
        
        flagged_users_summary = {}

        consistent_likes = user_metrics[
            (user_metrics['consistency_score'] < 0.1) & 
            (user_metrics['post_count'] > config.FRAUD_CONSISTENT_LIKES_MIN_POSTS) &
            (user_metrics['avg_likes'] > config.FRAUD_CONSISTENT_LIKES_MIN_AVG_LIKES)
        ]
        flagged_users_summary['Consistent Likes (Bot-like)'] = len(consistent_likes)
        if not consistent_likes.empty:
            report_lines.append("1. USERS WITH SUSPICIOUSLY CONSISTENT LIKES:")
            report_lines.append("    (Possible automated/bot behavior)")
            report_lines.append("-"*50)
            for username, row_data in consistent_likes.iterrows(): 
                report_lines.append(f"    @{username}: {row_data['avg_likes']:.0f} avg likes, "
                                  f"consistency score: {row_data['consistency_score']:.3f}, posts: {int(row_data['post_count'])}")
            report_lines.append("")
        
        like_fraud_suspects = user_metrics[
            (user_metrics['like_to_play_ratio'] > config.FRAUD_LIKE_TO_PLAY_RATIO_THRESHOLD) &
            (user_metrics['avg_likes'] > config.FRAUD_LIKE_MIN_AVG_LIKES)
        ]
        flagged_users_summary['Potential Like Fraud (Likes > Ratio x Video Plays)'] = len(like_fraud_suspects)
        if not like_fraud_suspects.empty:
            report_lines.append(f"2. POTENTIAL LIKE FRAUD (Likes > {config.FRAUD_LIKE_TO_PLAY_RATIO_THRESHOLD}x Video Plays):")
            report_lines.append("    (Suspicious like-to-play ratio)")
            report_lines.append("-"*50)
            for username, row_data in like_fraud_suspects.iterrows():
                report_lines.append(f"    @{username}: {row_data['avg_likes']:.0f} likes, "
                                  f"{row_data['avg_plays']:.0f} plays, "
                                  f"Ratio: {row_data['like_to_play_ratio']:.2f}")
            report_lines.append("")
        
        view_fraud_suspects = user_metrics[
            (user_metrics['play_to_like_ratio'] > config.FRAUD_PLAY_TO_LIKE_RATIO_THRESHOLD) &
            (user_metrics['avg_plays'] > config.FRAUD_PLAY_MIN_AVG_PLAYS)
        ]
        flagged_users_summary['Potential View Fraud (Video Plays > Ratio x Likes)'] = len(view_fraud_suspects)
        if not view_fraud_suspects.empty:
            report_lines.append(f"3. POTENTIAL VIEW FRAUD (Video Plays > {config.FRAUD_PLAY_TO_LIKE_RATIO_THRESHOLD}x Likes):")
            report_lines.append("    (Suspicious play-to-like ratio)")
            report_lines.append("-"*50)
            for username, row_data in view_fraud_suspects.iterrows():
                report_lines.append(f"    @{username}: {row_data['avg_plays']:.0f} plays, "
                                  f"{row_data['avg_likes']:.0f} likes, "
                                  f"Ratio: {row_data['play_to_like_ratio']:.2f}")
            report_lines.append("")
        
        spike_users = user_metrics[
            (user_metrics['avg_likes'] > 0) & 
            (user_metrics['like_spread'] > (user_metrics['avg_likes'] * 10))
        ]
        flagged_users_summary['Extreme Like Variations'] = len(spike_users)
        if not spike_users.empty:
            report_lines.append("4. USERS WITH EXTREME LIKE VARIATIONS:")
            report_lines.append("    (Possible purchased likes or viral anomalies)")
            report_lines.append("-"*50)
            for username, row_data in spike_users.iterrows():
                report_lines.append(f"    @{username}: Min {row_data['min_likes']:.0f}, "
                                  f"Max {row_data['max_likes']:.0f}, Spread: {row_data['like_spread']:.0f}")
            report_lines.append("")
        
        comment_fraud_suspects = user_metrics[
            (user_metrics['comment_to_play_ratio'] > config.FRAUD_COMMENT_TO_PLAY_RATIO_THRESHOLD) &
            (user_metrics['avg_comments'] > config.FRAUD_COMMENT_MIN_AVG_COMMENTS)
        ]
        flagged_users_summary['Potential Comment Fraud (Comments > Ratio x Video Plays)'] = len(comment_fraud_suspects)
        if not comment_fraud_suspects.empty:
            report_lines.append(f"5. POTENTIAL COMMENT FRAUD (Comments > {config.FRAUD_COMMENT_TO_PLAY_RATIO_THRESHOLD*100:.0f}% of Video Plays):")
            report_lines.append("    (Unusually high comment ratio)")
            report_lines.append("-"*50)
            for username, row_data in comment_fraud_suspects.iterrows():
                report_lines.append(f"    @{username}: {row_data['avg_comments']:.0f} comments, "
                                  f"{row_data['avg_plays']:.0f} plays, "
                                  f"Ratio: {row_data['comment_to_play_ratio']:.2f}")
            report_lines.append("")
        
        all_flagged_user_indices = set()
        if not consistent_likes.empty: all_flagged_user_indices.update(consistent_likes.index)
        if not like_fraud_suspects.empty: all_flagged_user_indices.update(like_fraud_suspects.index)
        if not view_fraud_suspects.empty: all_flagged_user_indices.update(view_fraud_suspects.index)
        if not spike_users.empty: all_flagged_user_indices.update(spike_users.index)
        if not comment_fraud_suspects.empty: all_flagged_user_indices.update(comment_fraud_suspects.index)
        total_suspicious = len(all_flagged_user_indices)
        
        report_lines.append("="*50)
        report_lines.append(f"📊 SUMMARY:")
        report_lines.append(f"Total users analyzed: {len(user_metrics)}")
        report_lines.append(f"Users with suspicious patterns: {total_suspicious}")
        percentage_suspicious = (total_suspicious / len(user_metrics) * 100) if len(user_metrics) > 0 else 0.0
        report_lines.append(f"Percentage suspicious: {percentage_suspicious:.1f}%")
        report_lines.append("")
        report_lines.append("🔍 BREAKDOWN BY FRAUD TYPE:")
        report_lines.append(f"- Consistent likes (bot-like): {flagged_users_summary.get('Consistent Likes (Bot-like)', 0)}")
        report_lines.append(f"- Like fraud suspects (Likes >> Video Plays): {flagged_users_summary.get('Potential Like Fraud (Likes > Ratio x Video Plays)', 0)}")
        report_lines.append(f"- View fraud suspects (Video Plays >> Likes): {flagged_users_summary.get('Potential View Fraud (Video Plays > Ratio x Likes)', 0)}")
        report_lines.append(f"- Extreme like variations: {flagged_users_summary.get('Extreme Like Variations', 0)}")
        report_lines.append(f"- Comment fraud suspects (Comments >> Video Plays): {flagged_users_summary.get('Potential Comment Fraud (Comments > Ratio x Video Plays)', 0)}")
        
        final_report_text = '\n'.join(report_lines) 
        report_path = os.path.join(self.isolation_forest_dir, config.FRAUD_SIGNALS_REPORT_FILE)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_report_text)
        
        print(f"✅ Fraud signal detection completed.")
        print(f"📄 Report saved to: {report_path}")
        
        return user_metrics, final_report_text, flagged_users_summary

    def create_visualizations(self):
        print("\n📊 Creating Main Analysis Dashboard...")
        if self.processed_df is None or self.processed_df.empty:
            print("❌ Cannot create main dashboard: Processed data is empty.")
            return None

        fig, axes = plt.subplots(2, 3, figsize=config.FIGURE_SIZE_MAIN_DASHBOARD) 
        main_dashboard_title = config.VISUALIZATIONS_FILE.replace('.png','').replace('_',' ').title()
        fig.suptitle(main_dashboard_title, fontsize=20, fontweight='bold')
        
        ax1 = axes[0, 0]
        if 'likesCount' in self.processed_df.columns:
            like_data = self.processed_df[self.processed_df['likesCount'] > 0]['likesCount']
            if not like_data.empty:
                ax1.hist(np.log10(like_data + 1), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                ax1.set_xlabel('Log10(Likes + 1)'); ax1.set_ylabel('Frequency'); ax1.set_title('Like Count Distribution')
                ax1.grid(True, alpha=0.3)
            else: ax1.text(0.5, 0.5, 'No like data > 0', ha='center', va='center', transform=ax1.transAxes)
        else: ax1.text(0.5, 0.5, 'likesCount missing', ha='center', va='center', transform=ax1.transAxes)

        ax2 = axes[0, 1]
        if 'videoDuration' in self.processed_df.columns and 'likesCount' in self.processed_df.columns:
            video_data = self.processed_df[(self.processed_df['videoDuration'] > 0) & (self.processed_df['likesCount'] > 0)]
            if not video_data.empty and len(video_data) > 10 :
                scatter = ax2.scatter(video_data['videoDuration'], np.log10(video_data['likesCount'] + 1), alpha=0.5, c=video_data['likesCount'], cmap='viridis', s=30)
                ax2.set_xlabel('Video Duration (s)'); ax2.set_ylabel('Log10(Likes + 1)'); ax2.set_title('Video Duration vs Likes')
                plt.colorbar(scatter, ax=ax2, label='Likes'); ax2.grid(True, alpha=0.3)
            else: ax2.text(0.5, 0.5, 'Not enough video data', ha='center', va='center', transform=ax2.transAxes)
        else: ax2.text(0.5, 0.5, 'videoDuration/likesCount missing', ha='center', va='center', transform=ax2.transAxes)
        
        ax3 = axes[0, 2]
        if 'hour' in self.processed_df.columns and not self.processed_df['hour'].empty and self.processed_df['hour'].iloc[0] != -1:
            hour_counts = self.processed_df['hour'].value_counts().sort_index()
            bars = ax3.bar(hour_counts.index, hour_counts.values, color='lightgreen', edgecolor='black')
            ax3.set_xlabel('Hour of Day'); ax3.set_ylabel('Number of Posts'); ax3.set_title('Posting Time Distribution')
            ax3.set_xticks(range(0, 24, 2)); ax3.grid(True, axis='y', alpha=0.3)
            for bar in bars: ax3.text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.05 * ax3.get_ylim()[1], f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8)
        else: ax3.text(0.5, 0.5, 'No hour data or column missing', ha='center', va='center', transform=ax3.transAxes)
        
        ax4 = axes[1, 0]
        if 'day_of_week' in self.processed_df.columns and 'likesCount' in self.processed_df.columns and not self.processed_df['day_of_week'].empty and self.processed_df['day_of_week'].iloc[0] != 'Unknown':
            day_order_en = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_order_fa_for_chart = [self._get_persian_display_text(d) for d in ['دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنج‌شنبه', 'جمعه', 'شنبه', 'یکشنبه']]
            
            day_map_for_chart = dict(zip(day_order_en, day_order_fa_for_chart))
            temp_df_day_week = self.processed_df.copy()
            temp_df_day_week['day_of_week_display'] = temp_df_day_week['day_of_week'].map(day_map_for_chart).fillna(temp_df_day_week['day_of_week'])


            day_likes = temp_df_day_week.groupby('day_of_week_display')['likesCount'].mean().reindex(day_order_fa_for_chart, fill_value=0)

            ax4.plot(range(len(day_likes)), day_likes.values, 'o-', color='purple', linewidth=2, markersize=8)
            ax4.set_xticks(range(len(day_likes))); ax4.set_xticklabels(day_likes.index) 
            ax4.set_ylabel('Average Likes'); ax4.set_title('Avg. Likes by Day of Week'); ax4.grid(True, alpha=0.3)
            for i, val in enumerate(day_likes.values): ax4.text(i, val + 0.02 * ax4.get_ylim()[1], f'{val:,.0f}', ha='center', va='bottom', fontsize=8)
        else: ax4.text(0.5, 0.5, 'day_of_week/likesCount missing', ha='center', va='center', transform=ax4.transAxes)
        
        nationality_titles = {
            'Iranian': self._get_persian_display_text('🇮🇷 ایرانی'), 
            'International': self._get_persian_display_text('🌍 بین‌المللی') 
        }

        for nat_ax, nationality_key, color_val in [(axes[1,1], 'Iranian', 'lightcoral'), (axes[1,2], 'International', 'lightblue')]:
            chart_title_suffix = nationality_titles.get(nationality_key, nationality_key) 
            if 'nationality' in self.processed_df.columns and 'ownerUsername' in self.processed_df.columns and 'likesCount' in self.processed_df.columns:
                nat_df = self.processed_df[self.processed_df['nationality'] == nationality_key]
                if not nat_df.empty:
                    nat_user_stats = nat_df.groupby('ownerUsername')['likesCount'].agg(['mean', 'count'])
                    nat_user_stats = nat_user_stats[nat_user_stats['count'] >= 3] 
                    top_nat_users = nat_user_stats.nlargest(10, 'mean')
                    if not top_nat_users.empty:
                        y_pos = np.arange(len(top_nat_users))
                        bars = nat_ax.barh(y_pos, top_nat_users['mean'], color=color_val, edgecolor='black')
                        
                        y_labels = [self._get_persian_display_text(str(user[:15]) + "..." if len(str(user)) > 15 else str(user)) for user in top_nat_users.index]
                        nat_ax.set_yticks(y_pos); nat_ax.set_yticklabels(y_labels, fontsize=9)
                        
                        nat_ax.set_xlabel('Average Likes'); nat_ax.set_title(f'Top 10 {chart_title_suffix} Users'); nat_ax.invert_yaxis(); nat_ax.grid(True, axis='x', alpha=0.3) 
                        for bar_obj in bars: nat_ax.text(bar_obj.get_width() + 0.01 * nat_ax.get_xlim()[1], bar_obj.get_y() + bar_obj.get_height()/2, f'{bar_obj.get_width():,.0f}', ha='left', va='center', fontsize=8)
                    else: nat_ax.text(0.5, 0.5, f'Not enough {nationality_key} user data', ha='center', va='center', transform=nat_ax.transAxes)
                else: nat_ax.text(0.5, 0.5, f'No {nationality_key} users found', ha='center', va='center', transform=nat_ax.transAxes)
            else: nat_ax.text(0.5, 0.5, 'Required columns missing', ha='center', va='center', transform=nat_ax.transAxes)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        main_dashboard_path = os.path.join(self.visualizations_dir, config.VISUALIZATIONS_FILE)
        plt.savefig(main_dashboard_path, dpi=config.DPI_SETTING, bbox_inches='tight')
        print(f"✅ Saved Main Analysis Dashboard to {main_dashboard_path}")
        plt.close(fig)

        self._create_overview_dashboard()
        self._create_anomaly_specific_visualization()
        self._create_eclat_pattern_visualization() 
        
        return True

    def _create_overview_dashboard(self):
        print("\n📊 Creating Overview Dashboard...")
        if self.processed_df is None or self.processed_df.empty:
            print("❌ Cannot create overview dashboard: Processed data is empty.")
            return

        fig, axes = plt.subplots(2, 2, figsize=config.FIGURE_SIZE_OVERVIEW)
        fig.suptitle('Instagram Content Overview Dashboard', fontsize=16, fontweight='bold')

        ax1 = axes[0, 0]
        if 'ownerUsername' in self.processed_df.columns and 'likesCount' in self.processed_df.columns:
            top_users = self.processed_df.groupby('ownerUsername')['likesCount'].sum().nlargest(config.TOP_N_DISPLAY)
            if not top_users.empty:
                y_labels = [self._get_persian_display_text(str(user)) for user in top_users.index] 
                bars1 = ax1.barh(range(len(y_labels)), top_users.values, color='skyblue')
                ax1.set_yticks(range(len(y_labels))); ax1.set_yticklabels(y_labels)
                ax1.set_xlabel('Total Likes'); ax1.set_title(f'Top {config.TOP_N_DISPLAY} Users by Total Likes'); ax1.invert_yaxis()
                for bar, value in zip(bars1, top_users.values): ax1.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(value):,}', va='center', ha='left', fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.5, pad=1))
            else: ax1.text(0.5, 0.5, 'No user data for top likes', ha='center', va='center', transform=ax1.transAxes)
        else: ax1.text(0.5, 0.5, 'ownerUsername/likesCount missing', ha='center', va='center', transform=ax1.transAxes)

        ax2 = axes[0, 1]
        if 'nationality' in self.processed_df.columns:
            nationality_dist = self.processed_df['nationality'].value_counts()
            if not nationality_dist.empty:
                colors = ['lightcoral', 'lightgreen', 'gold', 'lightskyblue', 'lightpink'] 
                pie_labels = [self._get_persian_display_text(str(label)) if label == 'Iranian' else str(label) for label in nationality_dist.index]
                wedges, texts, autotexts = ax2.pie(nationality_dist.values, labels=pie_labels, 
                                                   autopct='%1.1f%%', colors=colors[:len(nationality_dist)], startangle=90)
                ax2.set_title('Content Distribution by Nationality')
                for i, autotext in enumerate(autotexts): autotext.set_text(f"{nationality_dist.values[i]:,}\n({autotext.get_text()})")
            else: ax2.text(0.5, 0.5, 'No nationality data', ha='center', va='center', transform=ax2.transAxes)
        else: ax2.text(0.5, 0.5, 'nationality column missing', ha='center', va='center', transform=ax2.transAxes)

        ax3 = axes[1, 0]
        if 'like_category' in self.processed_df.columns:
            perf_dist = self.processed_df['like_category'].value_counts()
            if not perf_dist.empty:
                bar_labels = [str(label) for label in perf_dist.index.astype(str)] 
                bars3 = ax3.bar(bar_labels, perf_dist.values, color=['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCF7E', '#4ECDC4'][:len(perf_dist)])
                ax3.set_xticklabels(bar_labels, rotation=45, ha='right')
                ax3.set_ylabel('Number of Posts'); ax3.set_title('Posts by Performance Category')
                for bar, value in zip(bars3, perf_dist.values): ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*ax3.get_ylim()[1], f'{int(value):,}', ha='center', va='bottom', fontsize=9)
            else: ax3.text(0.5, 0.5, 'No performance category data', ha='center', va='center', transform=ax3.transAxes)
        else: ax3.text(0.5, 0.5, 'like_category column missing', ha='center', va='center', transform=ax3.transAxes)
        
        ax4 = axes[1, 1]
        if 'hashtags' in self.processed_df.columns:
            all_hashtags_list = [] 
            for tags_str in self.processed_df['hashtags'].dropna(): 
                extracted_tags = self.extract_hashtags(tags_str) 
                all_hashtags_list.extend(extracted_tags)
            
            if all_hashtags_list:
                hashtag_counts = Counter(all_hashtags_list).most_common(config.TOP_N_DISPLAY)
                if hashtag_counts:
                    tags, counts = zip(*hashtag_counts)
                    y_labels = [self._get_persian_display_text(str(tag)) for tag in tags] 
                    bars4 = ax4.barh(range(len(y_labels)), counts, color='mediumpurple')
                    ax4.set_yticks(range(len(y_labels))); ax4.set_yticklabels(y_labels)
                    ax4.set_xlabel('Frequency'); ax4.set_title(f'Top {config.TOP_N_DISPLAY} Most Used Hashtags'); ax4.invert_yaxis()
                    for bar, value in zip(bars4, counts): ax4.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(value):,}', va='center', ha='left', fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.5, pad=1))
                else: ax4.text(0.5, 0.5, 'No hashtags found', ha='center', va='center', transform=ax4.transAxes)
            else: ax4.text(0.5, 0.5, 'No hashtags to count', ha='center', va='center', transform=ax4.transAxes)
        else: ax4.text(0.5, 0.5, 'hashtags column missing', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        overview_dashboard_path = os.path.join(self.visualizations_dir, config.OVERVIEW_DASHBOARD_FILE)
        plt.savefig(overview_dashboard_path, dpi=config.DPI_SETTING, bbox_inches='tight')
        print(f"✅ Saved Overview Dashboard to {overview_dashboard_path}")
        plt.close(fig)

    def _create_anomaly_specific_visualization(self):
        print("\n📊 Creating Anomaly-Specific Visualizations...")
        required_cols = ['is_anomaly', 'engagement_rate', 'likesCount', 'ownerUsername'] 
        
        if self.processed_df is None or self.processed_df.empty:
            print("❌ Cannot create anomaly visualization: Processed data is empty.")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.FIGURE_SIZE_ANOMALY_SPECIFIC)
            fig.suptitle('Anomaly Detection Insights', fontsize=16, fontweight='bold')
            ax1.text(0.5, 0.5, "Processed data is empty", ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, "Processed data is empty", ha='center', va='center', transform=ax2.transAxes)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            anomaly_vis_path = os.path.join(self.visualizations_dir, config.ANOMALY_SPECIFIC_VIS_FILE)
            plt.savefig(anomaly_vis_path, dpi=config.DPI_SETTING, bbox_inches='tight')
            plt.close(fig)
            return

        missing_cols = [col for col in required_cols if col not in self.processed_df.columns]
        if missing_cols:
            print(f"❌ Cannot create anomaly visualization: Missing required columns in processed_df: {', '.join(missing_cols)}. Ensure anomaly detection ran and added these columns.")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.FIGURE_SIZE_ANOMALY_SPECIFIC)
            fig.suptitle('Anomaly Detection Insights', fontsize=16, fontweight='bold')
            ax1.text(0.5, 0.5, f"Missing: {', '.join(missing_cols)}", ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, f"Missing: {', '.join(missing_cols)}", ha='center', va='center', transform=ax2.transAxes) 
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            anomaly_vis_path = os.path.join(self.visualizations_dir, config.ANOMALY_SPECIFIC_VIS_FILE)
            plt.savefig(anomaly_vis_path, dpi=config.DPI_SETTING, bbox_inches='tight')
            plt.close(fig)
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.FIGURE_SIZE_ANOMALY_SPECIFIC)
        fig.suptitle('Anomaly Detection Insights', fontsize=16, fontweight='bold')

        normal_posts = self.processed_df[self.processed_df['is_anomaly'] == False]
        anomaly_posts = self.processed_df[self.processed_df['is_anomaly'] == True]
            
        if not normal_posts.empty or not anomaly_posts.empty:
            if not normal_posts.empty:
                ax1.scatter(normal_posts['likesCount'], normal_posts['engagement_rate'], alpha=0.5, s=30, c='blue', label='Normal') 
            if not anomaly_posts.empty:
                ax1.scatter(anomaly_posts['likesCount'], anomaly_posts['engagement_rate'], alpha=0.7, s=50, c='red', label='Anomaly', edgecolors='black') 
            
            ax1.set_xlabel('Likes Count (Log Scale)'); ax1.set_ylabel('Engagement Rate') 
            ax1.set_title('Likes vs Engagement Rate (Anomalies Highlighted)'); 
            if not normal_posts.empty or not anomaly_posts.empty : ax1.legend() 
            ax1.set_xscale('log') 
            min_likes_for_plot = 1
            if 'likesCount' in self.processed_df and not self.processed_df[self.processed_df['likesCount'] > 0]['likesCount'].empty:
                min_likes_for_plot = max(1, self.processed_df[self.processed_df['likesCount'] > 0]['likesCount'].min())
            ax1.set_xlim(left=min_likes_for_plot)
            ax1.set_ylim(bottom=0) 
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No data to plot for likes vs engagement', ha='center', va='center', transform=ax1.transAxes)
        
        if not anomaly_posts.empty:
            anomaly_users = anomaly_posts.groupby('ownerUsername').size().nlargest(config.TOP_N_DISPLAY)
            if not anomaly_users.empty:
                y_labels = [self._get_persian_display_text(str(user)) for user in anomaly_users.index] 
                bars = ax2.barh(range(len(y_labels)), anomaly_users.values, color='salmon')
                ax2.set_yticks(range(len(y_labels))); ax2.set_yticklabels(y_labels)
                ax2.set_xlabel('Number of Anomalous Posts'); ax2.set_title(f'Top {config.TOP_N_DISPLAY} Users with Most Anomalous Posts'); ax2.invert_yaxis()
                for bar, value in zip(bars, anomaly_users.values): ax2.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{int(value)}', va='center', ha='left', fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.5, pad=1))
            else:
                 ax2.text(0.5, 0.5, 'No users with anomalous posts to display', ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, 'No anomalous posts data for user plot', ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        anomaly_vis_path = os.path.join(self.visualizations_dir, config.ANOMALY_SPECIFIC_VIS_FILE)
        plt.savefig(anomaly_vis_path, dpi=config.DPI_SETTING, bbox_inches='tight')
        print(f"✅ Saved Anomaly-Specific Visualizations to {anomaly_vis_path}")
        plt.close(fig)

    def _create_eclat_pattern_visualization(self):
        print("\n📊 Creating Eclat Pattern Visualizations...")
        
        eclat_df_to_plot = self.hashtag_patterns_df 
        main_plot_title_for_chart = self._get_persian_display_text("الگوهای برتر هشتگ Eclat بر اساس اندازه مجموعه آیتم")
        filename = config.ECLAT_PATTERNS_VIS_FILE

        if eclat_df_to_plot is None or eclat_df_to_plot.empty or 'size' not in eclat_df_to_plot.columns:
            print(f"❌ Cannot create Eclat pattern visualization for '{filename}': Data or 'size' column is empty/missing.")
            return

        unique_sizes = sorted(eclat_df_to_plot['size'].unique())
        num_sizes = len(unique_sizes)

        if num_sizes == 0:
            print(f"❌ No Eclat patterns with size information found for '{filename}'.")
            return

        ncols = 2 if num_sizes > 1 else 1
        nrows = (num_sizes + ncols - 1) // ncols 
        
        fig_height = max(6, nrows * 4.5) 
        fig_width = config.FIGURE_SIZE_PATTERNS_SPECIFIC[0] if ncols > 1 else config.FIGURE_SIZE_PATTERNS_SPECIFIC[0] / 1.5

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False) 
        fig.suptitle(main_plot_title_for_chart, fontsize=16, fontweight='bold') 
        axes = axes.flatten() 

        plot_idx = 0
        for size_val in unique_sizes:
            if plot_idx >= len(axes): 
                break
            
            ax = axes[plot_idx]
            size_patterns = eclat_df_to_plot[eclat_df_to_plot['size'] == size_val].sort_values('support', ascending=False).head(config.TOP_N_DISPLAY)
            
            if not size_patterns.empty:
                y_labels = [self._get_persian_display_text(str(item)) for item in size_patterns['itemset']]
                bars = ax.barh(range(len(y_labels)), size_patterns['support'] * 100, color=f'C{plot_idx % 10}') 
                ax.set_yticks(range(len(y_labels)))
                ax.set_yticklabels(y_labels, fontsize=8)
                
                ax.set_xlabel('Support (%)') 
                subplot_title = self._get_persian_display_text(f'{min(config.TOP_N_DISPLAY, len(size_patterns))} الگوی برتر با اندازه {size_val}')
                ax.set_title(subplot_title)
                ax.invert_yaxis()
                for bar, value in zip(bars, size_patterns['support']*100): 
                    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{value:.2f}%', va='center', ha='left', fontsize=8)
            else:
                ax.text(0.5,0.5, f"No patterns of size {size_val}", ha='center', va='center', transform=ax.transAxes)
            plot_idx += 1
        
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        eclat_patterns_path = os.path.join(self.visualizations_dir, filename)
        plt.savefig(eclat_patterns_path, dpi=config.DPI_SETTING, bbox_inches='tight')
        print(f"✅ Saved Dynamic Eclat Pattern Visualization to {eclat_patterns_path}")
        plt.close(fig)

    def generate_main_report(self, hashtag_results_df, user_results_df, temporal_results_df, anomaly_reports_summary_dict):
        report = []
        report_title = config.MAIN_ECLAT_REPORT_FILE.replace('.txt','').replace('_',' ').upper() 
        report.append("="*70 + f"\n📊 {report_title} ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n" + "="*70)
        
        report.append("\n📈 GENERAL STATISTICS:") 
        report.append(f"• Excel File: {self.excel_file}, Sheet: {self.sheet_name}") 
        report.append(f"• Total posts loaded: {len(self.df) if self.df is not None else 'N/A'}") 
        report.append(f"• Posts processed (likes > 0): {len(self.processed_df) if self.processed_df is not None else 'N/A'}") 
        
        unique_users_count = 'N/A'
        if self.df is not None and 'ownerUsername' in self.df.columns:
            unique_users_count = self.df['ownerUsername'].nunique()
        report.append(f"• Unique users: {unique_users_count}") 
        
        if self.processed_df is not None and 'nationality' in self.processed_df.columns:
            iranian_count = (self.processed_df['nationality'] == 'Iranian').sum()
            intl_count = (self.processed_df['nationality'] == 'International').sum()
            total_processed = len(self.processed_df) if self.processed_df is not None else 0
            report.append(f"\n🌍 NATIONALITY BREAKDOWN (Processed Posts):") 
            if total_processed > 0:
                report.append(f"• Iranian posts: {iranian_count} ({((iranian_count/total_processed*100)):.1f}%)") 
                report.append(f"• International posts: {intl_count} ({((intl_count/total_processed*100)):.1f}%)") 
            else:
                report.append(f"• Iranian posts: {iranian_count} (0.0%)") 
                report.append(f"• International posts: {intl_count} (0.0%)") 
        
        report.append("\n\n" + "="*70 + "\n🔍 ECLAT PATTERN ANALYSIS SUMMARY (Top 5 from each category)\n" + "="*70) 
        report.append(f"(Min Support: {self.min_support*100:.1f}%, Max Itemset Size: {self.max_eclat_itemset_size})") 

        eclat_data_map = {
            "Hashtag": (hashtag_results_df, "posts"), 
            "User": (user_results_df, "users"),       
            "Temporal": (temporal_results_df, "posts") 
        }

        for name, (results_df, unit) in eclat_data_map.items():
            report.append(f"\n📌 TOP {name.upper()} PATTERNS:") 
            if results_df is not None and not results_df.empty:
                top_5 = results_df.sort_values('support', ascending=False).head(5)
                for idx, row in top_5.iterrows():
                    report.append(f"  - Itemset: {row['itemset']} (Size: {row['size']})") 
                    report.append(f"    Support: {row['support']:.2%} ({row['count']} {unit})") 
            else: report.append(f"  No significant {name.lower()} patterns found.") 
        report.append(f"\n(Detailed Eclat patterns saved by size in: '{self.eclat_patterns_parent_dir}')") 

        report.append("\n\n" + "="*70 + "\n🚨 ANOMALY DETECTION SUMMARY (from detailed reports)\n" + "="*70) 
        for anomaly_type_key, result_tuple in anomaly_reports_summary_dict.items():
            report.append(f"\n--- {anomaly_type_key.replace('_', ' ').title()} ---") 
            if result_tuple and len(result_tuple) >= 2 and result_tuple[1]:
                text_report_content = result_tuple[1] 
                summary_lines_extracted = []
                lines_count = 0
                # Extract summary lines from the beginning of each detailed anomaly report
                # Look for lines like "Total ... analyzed:", "Anomalies (general) detected:", 
                # "Positive Anomalies:", "Negative Anomalies:", specific user/content anomaly types.
                in_summary_block = True
                for line in text_report_content.split('\n'):
                    stripped_line = line.strip()
                    if not stripped_line: continue

                    # Heuristic to identify end of summary block in detailed reports
                    if stripped_line.startswith("="*70) and lines_count > 0 and "REPORT" not in stripped_line : # End of header, start of detailed list
                        if any(cat_title in stripped_line for cat_title in ["🌟 Positive Anomalies", "📉 Negative Anomalies", "👤 Users with", "🤖 Users with", "📱 Unusual Content", "📄 Other General"]):
                             in_summary_block = False # Moved to detailed listing part

                    if in_summary_block and not stripped_line.startswith("=") and not stripped_line.startswith("-") and \
                       ("Total posts analyzed:" in stripped_line or \
                        "Total users analyzed:" in stripped_line or \
                        "Anomalies (general) detected:" in stripped_line or 
                        "Positive Anomalies:" in stripped_line or \
                        "Negative Anomalies:" in stripped_line or \
                        "Users with Low Activity & High Engagement" in stripped_line or \
                        "Users with 100% Viral Posts" in stripped_line or \
                        "Unusual Content:" in stripped_line or \
                        (anomaly_type_key == "Fraud Signals" and "USERS WITH SUSPICIOUSLY" in stripped_line.upper()) or \
                        (anomaly_type_key == "Fraud Signals" and "SUMMARY:" in stripped_line.upper()) or \
                        (anomaly_type_key == "Fraud Signals" and "BREAKDOWN BY FRAUD TYPE:" in stripped_line.upper()) or \
                        (anomaly_type_key == "Fraud Signals" and stripped_line.startswith("- ")) # For fraud breakdown list items
                       ): 
                        summary_lines_extracted.append(f"  {stripped_line}")
                        lines_count += 1
                    
                    if lines_count >= 10 and anomaly_type_key != "Fraud Signals": break # Limit lines for brevity, except for fraud
                    if lines_count >= 15 and anomaly_type_key == "Fraud Signals": break


                if summary_lines_extracted:
                    report.extend(summary_lines_extracted)
                else: 
                    report.append("  (See detailed report for specifics or no key summary lines found in detailed report)") 
            else:
                report.append("  (No report data available or error during generation)") 
        report.append(f"\n(Detailed anomaly reports saved in: '{self.isolation_forest_dir}')") 
        
        report_text = '\n'.join(report)
        report_path = os.path.join(self.eclat_reports_dir, config.MAIN_ECLAT_REPORT_FILE)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n📄 Main Eclat report saved to: {report_path}") 
        return report_text

    def generate_summary_file(self, all_results_data):
        summary_lines = []
        summary_lines.append("======================================================================")
        report_date_str = f"{self.timestamp_str[:4]}-{self.timestamp_str[4:6]}-{self.timestamp_str[6:8]} {self.timestamp_str[9:11]}:{self.timestamp_str[11:13]}"
        summary_lines.append(f"📊 {config.MAIN_ECLAT_REPORT_FILE.replace('.txt','').replace('_',' ').upper()} ({report_date_str})")
        summary_lines.append("======================================================================")

        summary_lines.append("\n📈 GENERAL STATISTICS:")
        summary_lines.append(f"• Excel File: {self.excel_file}, Sheet: {self.sheet_name}")
        total_loaded = len(self.df) if self.df is not None else 'N/A'
        total_processed = len(self.processed_df) if self.processed_df is not None and not self.processed_df.empty else 0
        summary_lines.append(f"• Total posts loaded: {total_loaded}")
        summary_lines.append(f"• Posts processed (likes > 0): {total_processed if total_processed > 0 else 'N/A'}")
        
        unique_users = 'N/A'
        if self.df is not None and 'ownerUsername' in self.df.columns: 
            unique_users = self.df['ownerUsername'].nunique()
        summary_lines.append(f"• Unique users: {unique_users}")

        if self.processed_df is not None and not self.processed_df.empty and 'nationality' in self.processed_df.columns:
            iranian_count = (self.processed_df['nationality'] == 'Iranian').sum()
            intl_count = (self.processed_df['nationality'] == 'International').sum()
            summary_lines.append(f"\n🌍 NATIONALITY BREAKDOWN (Processed Posts):")
            if total_processed > 0: 
                summary_lines.append(f"• Iranian posts: {iranian_count} ({ (iranian_count/total_processed*100) :.1f}%)")
                summary_lines.append(f"• International posts: {intl_count} ({ (intl_count/total_processed*100) :.1f}%)")
            else:
                 summary_lines.append(f"• Iranian posts: {iranian_count} (N/A%)")
                 summary_lines.append(f"• International posts: {intl_count} (N/A%)")
        else:
            summary_lines.append("\n🌍 NATIONALITY BREAKDOWN (Processed Posts): Data unavailable or not applicable.")

        summary_lines.append("\n\n======================================================================")
        summary_lines.append(f"🔍 ECLAT PATTERN ANALYSIS SUMMARY (Top 5 from each category)")
        summary_lines.append("======================================================================")
        summary_lines.append(f"(Min Support: {self.min_support*100:.1f}%, Max Itemset Size: {self.max_eclat_itemset_size})")

        eclat_dfs = {
            "HASHTAGS": (all_results_data.get('hashtag_patterns'), "posts"),
            "USERS": (all_results_data.get('user_patterns'), "users"),
            "TEMPORAL": (all_results_data.get('temporal_patterns'), "posts")
        }
        for name, (df, unit) in eclat_dfs.items():
            summary_lines.append(f"\n📌 TOP {name} PATTERNS:")
            if df is not None and not df.empty:
                top_5 = df.sort_values('support', ascending=False).head(5)
                for i, (_, row) in enumerate(top_5.iterrows(), 1):
                    summary_lines.append(f"  {i}. Itemset: \"{row['itemset']}\" (Size: {row['size']})")
                    summary_lines.append(f"      Support: {row['support']:.2%} ({row['count']} {unit})")
            else:
                summary_lines.append("      No significant patterns found with current settings.")
        summary_lines.append(f"\n(Detailed Eclat patterns saved by size in: '{self.eclat_patterns_parent_dir}')")

        summary_lines.append("\n\n======================================================================")
        summary_lines.append(f"🚨 ANOMALY DETECTION SUMMARY (Top 5 Highlights & Overall Counts)")
        summary_lines.append("======================================================================")
        anomaly_summary_dict = all_results_data.get('anomaly_reports_summary', {})

        # Anomalous Posts
        summary_lines.append("\n--- 🚀 Anomalous Posts ---")
        if 'Anomalous Posts' in anomaly_summary_dict and anomaly_summary_dict['Anomalous Posts'] and len(anomaly_summary_dict['Anomalous Posts']) == 3:
            posts_df, _, posts_specific_findings = anomaly_summary_dict['Anomalous Posts']
            
            total_posts_analyzed_for_summary = len(self.processed_df) if self.processed_df is not None and not self.processed_df.empty else 0
            
            num_anomalous_posts = 0
            percent_anomalous_posts = 0.0
            if posts_df is not None and not posts_df.empty:
                num_anomalous_posts = len(posts_df)
                if total_posts_analyzed_for_summary > 0:
                    percent_anomalous_posts = (num_anomalous_posts / total_posts_analyzed_for_summary) * 100
            
            summary_lines.append(f"  Total posts analyzed: {total_posts_analyzed_for_summary if total_posts_analyzed_for_summary > 0 else 'N/A'}")
            summary_lines.append(f"  Anomalies (general) detected: {num_anomalous_posts} ({percent_anomalous_posts:.1f}%)")

            if posts_specific_findings:
                for category, findings_df in posts_specific_findings.items():
                    if findings_df is not None and not findings_df.empty:
                        summary_lines.append(f"  🎯 Top 5 for '{category.split(' - ')[-1]}': ({len(findings_df)} found)") # Show count here
                        for i, (_, row) in enumerate(findings_df.head(5).iterrows(), 1): 
                            owner = row.get('ownerUsername','N/A') if pd.notna(row.get('ownerUsername')) else 'N/A'
                            likes = row.get('likesCount',0) if pd.notna(row.get('likesCount')) else 0
                            score = row.get('anomaly_score',0) if pd.notna(row.get('anomaly_score')) else 0
                            summary_lines.append(f"    {i}. @{owner} (Likes: {likes:,.0f}, Score: {score:.3f})")
            elif posts_df is not None and not posts_df.empty : 
                summary_lines.append(f"  🏆 Top 5 Most Anomalous Posts (general, by score):")
                for i, (_, row) in enumerate(posts_df.head(5).iterrows(), 1):
                    owner = row.get('ownerUsername','N/A') if pd.notna(row.get('ownerUsername')) else 'N/A'
                    likes = row.get('likesCount',0) if pd.notna(row.get('likesCount')) else 0
                    score = row.get('anomaly_score',0) if pd.notna(row.get('anomaly_score')) else 0
                    summary_lines.append(f"    {i}. @{owner} (Likes: {likes:,.0f}, Score: {score:.3f})")
            else: summary_lines.append("    No specific anomalous post categories highlighted or data unavailable.")
        else: summary_lines.append("    Anomalous Posts data unavailable.")

        # Anomalous Users
        summary_lines.append("\n--- 👤 Anomalous Users ---")
        if 'Anomalous Users' in anomaly_summary_dict and anomaly_summary_dict['Anomalous Users'] and len(anomaly_summary_dict['Anomalous Users']) == 3:
            users_df, _, users_specific_findings = anomaly_summary_dict['Anomalous Users']
            
            total_users_analyzed_for_summary = 0
            if self.processed_df is not None and not self.processed_df.empty and 'ownerUsername' in self.processed_df.columns:
                 total_users_analyzed_for_summary = self.processed_df['ownerUsername'].nunique()

            num_anomalous_users = 0
            percent_anomalous_users = 0.0
            if users_df is not None and not users_df.empty:
                num_anomalous_users = len(users_df)
                if total_users_analyzed_for_summary > 0:
                    percent_anomalous_users = (num_anomalous_users / total_users_analyzed_for_summary) * 100
            
            summary_lines.append(f"  Total users analyzed: {total_users_analyzed_for_summary if total_users_analyzed_for_summary > 0 else 'N/A'}")
            summary_lines.append(f"  Anomalies (general) detected: {num_anomalous_users} ({percent_anomalous_users:.1f}%)")

            if users_specific_findings:
                for category, findings_df in users_specific_findings.items():
                    if findings_df is not None and not findings_df.empty:
                        summary_lines.append(f"  🎯 Top 5 for '{category.split(' - ')[-1]}': ({len(findings_df)} found)")
                        for i, (username, row) in enumerate(findings_df.head(5).iterrows(), 1):
                            avg_likes = row.get('avg_likes',0) if pd.notna(row.get('avg_likes')) else 0
                            score = row.get('anomaly_score',0) if pd.notna(row.get('anomaly_score')) else 0
                            summary_lines.append(f"    {i}. @{username} (Avg. Likes: {avg_likes:,.0f}, Score: {score:.3f})")
            elif users_df is not None and not users_df.empty:
                summary_lines.append(f"  🏆 Top 5 Most Anomalous Users (general, by score):")
                for i, (username, row) in enumerate(users_df.head(5).iterrows(), 1):
                    avg_likes = row.get('avg_likes',0) if pd.notna(row.get('avg_likes')) else 0
                    score = row.get('anomaly_score',0) if pd.notna(row.get('anomaly_score')) else 0
                    summary_lines.append(f"    {i}. @{username} (Avg. Likes: {avg_likes:,.0f}, Score: {score:.3f})")
            else: summary_lines.append("    No specific anomalous user categories highlighted or data unavailable.")
        else: summary_lines.append("    Anomalous Users data unavailable.")
            
        # Content Anomalies
        summary_lines.append("\n--- 📄 Content Anomalies ---")
        if 'Content Anomalies' in anomaly_summary_dict and anomaly_summary_dict['Content Anomalies'] and len(anomaly_summary_dict['Content Anomalies']) == 3:
            content_df, _, content_specific_findings = anomaly_summary_dict['Content Anomalies']

            total_content_analyzed_for_summary = len(self.processed_df) if self.processed_df is not None and not self.processed_df.empty else 0
            
            num_anomalous_content = 0
            percent_anomalous_content = 0.0
            if content_df is not None and not content_df.empty:
                num_anomalous_content = len(content_df)
                if total_content_analyzed_for_summary > 0:
                    percent_anomalous_content = (num_anomalous_content / total_content_analyzed_for_summary) * 100

            summary_lines.append(f"  Total posts analyzed for content: {total_content_analyzed_for_summary if total_content_analyzed_for_summary > 0 else 'N/A'}")
            summary_lines.append(f"  Content Anomalies (general) detected: {num_anomalous_content} ({percent_anomalous_content:.1f}%)")

            if content_specific_findings:
                for category, findings_df in content_specific_findings.items():
                    if findings_df is not None and not findings_df.empty:
                        summary_lines.append(f"  🎯 Top 5 for '{category.split(' - ')[-1]}': ({len(findings_df)} found)")
                        for i, (_, row) in enumerate(findings_df.head(5).iterrows(), 1):
                            owner = row.get('ownerUsername','N/A') if pd.notna(row.get('ownerUsername')) else 'N/A'
                            likes = row.get('likesCount',0) if pd.notna(row.get('likesCount')) else 0
                            score_col_name = 'content_anomaly_score' if 'content_anomaly_score' in row else 'anomaly_score'
                            score = row.get(score_col_name,0) if pd.notna(row.get(score_col_name)) else 0
                            summary_lines.append(f"    {i}. Content by @{owner} (Likes: {likes:,.0f}, Score: {score:.3f})")
            elif content_df is not None and not content_df.empty:
                summary_lines.append(f"  🏆 Top 5 Most Anomalous Content (general, by score):")
                for i, (_, row) in enumerate(content_df.head(5).iterrows(), 1):
                    owner = row.get('ownerUsername','N/A') if pd.notna(row.get('ownerUsername')) else 'N/A'
                    likes = row.get('likesCount',0) if pd.notna(row.get('likesCount')) else 0
                    score_col_name = 'content_anomaly_score' if 'content_anomaly_score' in row else 'anomaly_score'
                    score = row.get(score_col_name,0) if pd.notna(row.get(score_col_name)) else 0
                    summary_lines.append(f"    {i}. Content by @{owner} (Likes: {likes:,.0f}, Score: {score:.3f})")
            else: summary_lines.append("    No specific content anomaly categories highlighted or data unavailable.")
        else: summary_lines.append("    Content Anomalies data unavailable.")

        # Fraud Signals
        summary_lines.append("\n--- 🛡️ Fraud Signals Summary ---")
        if 'Fraud Signals' in anomaly_summary_dict and anomaly_summary_dict['Fraud Signals'] and len(anomaly_summary_dict['Fraud Signals']) == 3:
            _ , report_text, flagged_counts = anomaly_summary_dict['Fraud Signals'] 
            if flagged_counts and any(value > 0 for value in flagged_counts.values() if isinstance(value, (int, float))): 
                summary_lines.append("  Users flagged for potential fraudulent activity patterns:")
                for pattern_key, count in flagged_counts.items():
                    if isinstance(count, (int,float)) and count > 0: 
                        summary_lines.append(f"    - {pattern_key}: {count} users.")
                
                most_prevalent_fraud_type = ""
                max_count = -1
                for pattern, count_val in flagged_counts.items(): 
                    if isinstance(count_val, int) and count_val > max_count: 
                        max_count = count_val
                        most_prevalent_fraud_type = pattern
                
                if max_count > 0:
                    summary_lines.append(f"    (Most prevalent: '{most_prevalent_fraud_type}' with {max_count} users.)")

            elif ("No users flagged" in report_text or "هیچ کاربری برای الگوهای سیگنال تقلب تعریف شده پرچم‌گذاری نشد" in report_text) or \
                 (flagged_counts and not any(value > 0 for value in flagged_counts.values() if isinstance(value, (int, float)))):
                summary_lines.append("    No users flagged for defined fraud signal patterns based on current heuristics.")
            else:
                summary_lines.append("    Fraud signal analysis performed. See detailed report for specifics.")
        else:
            summary_lines.append("    Fraud Signals data unavailable or error during generation.")
        summary_lines.append(f"\n(Detailed anomaly reports saved in: '{self.isolation_forest_dir}')")


        summary_lines.append("\n\n" + "="*30 + " 📂 OUTPUT FILE LOCATIONS " + "="*30)
        summary_lines.append(f"  Base Directory: {self.base_output_dir}")
        summary_lines.append(f"  ECLAT Patterns (by size): {self.eclat_patterns_parent_dir}/<pattern_type>/patterns_size_X.txt")
        summary_lines.append(f"  Main ECLAT Report: {os.path.join(self.eclat_reports_dir, config.MAIN_ECLAT_REPORT_FILE)}")
        summary_lines.append(f"  Anomaly Detail Reports: {self.isolation_forest_dir}")
        summary_lines.append(f"  Visualizations (Main Dashboard): {os.path.join(self.visualizations_dir, config.VISUALIZATIONS_FILE)}")
        summary_lines.append(f"  Visualizations (Overview Dashboard): {os.path.join(self.visualizations_dir, config.OVERVIEW_DASHBOARD_FILE)}")
        summary_lines.append(f"  Visualizations (Anomaly Specific Plots): {os.path.join(self.visualizations_dir, config.ANOMALY_SPECIFIC_VIS_FILE)}")
        summary_lines.append(f"  Visualizations (Eclat Patterns Plots): {os.path.join(self.visualizations_dir, config.ECLAT_PATTERNS_VIS_FILE)}")
        
        summary_report_filename = f"{config.SUMMARY_REPORT_FILE.split('.')[0]}_{self.timestamp_str}.txt"
        summary_report_path = os.path.join(self.base_output_dir, summary_report_filename)
        with open(summary_report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary_lines))
        print(f"\n📄 Overall summary report saved to: {summary_report_path}")
        return "\n".join(summary_lines)

    def run_complete_analysis(self):
        print("\n🚀 Starting complete analysis pipeline...")
        print("="*70)
        
        self._initialize_output_directories()
        
        if self.load_data() is None: 
            print("❌ Data loading failed. Aborting analysis.")
            return None
        
        if self.df is None or self.df.empty: 
            print("❌ No data available after attempting to load. Aborting analysis.")
            return None
        
        self.preprocess_data()
        if self.processed_df is None or self.processed_df.empty:
            print("❌ No data after preprocessing. Aborting analysis.")
            return None

        print("\n--- Running ECLAT Analyses ---")
        try: self.hashtag_patterns_df = self.analyze_hashtag_patterns()
        except Exception as e: print(f"❌ Error in hashtag Eclat analysis: {e}")
        
        try: self.user_patterns_df = self.analyze_user_patterns()
        except Exception as e: print(f"❌ Error in user Eclat analysis: {e}")

        try: self.temporal_patterns_df = self.analyze_temporal_patterns()
        except Exception as e: print(f"❌ Error in temporal Eclat analysis: {e}")
        
        print("\n\n--- Running Anomaly Detection Algorithms ---")
        anomaly_reports_summary_dict = {} 
        
        try: 
            df, text, specific_summary = self.detect_anomalous_posts()
            anomaly_reports_summary_dict['Anomalous Posts'] = (df,text, specific_summary)
        except Exception as e: 
            print(f"❌ Error in anomalous posts detection: {e}")
            anomaly_reports_summary_dict['Anomalous Posts'] = (pd.DataFrame(), f"Error: {e}", {})

        try: 
            df, text, specific_summary = self.detect_anomalous_users()
            anomaly_reports_summary_dict['Anomalous Users'] = (df,text, specific_summary)
        except Exception as e: 
            print(f"❌ Error in anomalous users detection: {e}")
            anomaly_reports_summary_dict['Anomalous Users'] = (pd.DataFrame(), f"Error: {e}", {})

        try: 
            df, text, specific_summary = self.detect_content_anomalies()
            anomaly_reports_summary_dict['Content Anomalies'] = (df,text, specific_summary)
        except Exception as e: 
            print(f"❌ Error in content anomalies detection: {e}")
            anomaly_reports_summary_dict['Content Anomalies'] = (pd.DataFrame(), f"Error: {e}", {})

        try: 
            fraud_df, fraud_text, fraud_flagged_summary = self.detect_fraud_signals()
            anomaly_reports_summary_dict['Fraud Signals'] = (fraud_df, fraud_text, fraud_flagged_summary)
        except Exception as e: 
            print(f"❌ Error in fraud signals detection: {e}")
            anomaly_reports_summary_dict['Fraud Signals'] = (pd.DataFrame(), f"Error: {e}", {})
        
        print("\n\n--- Generating Reports and Visualizations ---")
        main_report_text = self.generate_main_report(self.hashtag_patterns_df, self.user_patterns_df, self.temporal_patterns_df, anomaly_reports_summary_dict)
        
        try:
            self.create_visualizations()
        except Exception as e:
            print(f"❌ Error creating/saving visualizations: {e}")
        
        all_results_data = {
            'hashtag_patterns': self.hashtag_patterns_df,
            'user_patterns': self.user_patterns_df,
            'temporal_patterns': self.temporal_patterns_df,
            'anomaly_reports_summary': anomaly_reports_summary_dict, 
            'main_report_text': main_report_text
        }
        self.generate_summary_file(all_results_data)

        print("\n\n✅ Analysis completed!")
        print("="*70)
        print(f"📊 All results saved in directory: {self.base_output_dir}")
        summary_report_filename = f"{config.SUMMARY_REPORT_FILE.split('.')[0]}_{self.timestamp_str}.txt"
        print(f"  - Comprehensive Summary Report: {os.path.join(self.base_output_dir, summary_report_filename)}")
        print(f"  - Main ECLAT & Anomaly Report: {os.path.join(self.eclat_reports_dir, config.MAIN_ECLAT_REPORT_FILE)}")
        print(f"  - Visualizations Dashboard: {os.path.join(self.visualizations_dir, config.VISUALIZATIONS_FILE)}")
        print(f"  - Overview Dashboard: {os.path.join(self.visualizations_dir, config.OVERVIEW_DASHBOARD_FILE)}")
        print(f"  - Anomaly Specific Plots: {os.path.join(self.visualizations_dir, config.ANOMALY_SPECIFIC_VIS_FILE)}")
        print(f"  - Eclat Patterns Plots: {os.path.join(self.visualizations_dir, config.ECLAT_PATTERNS_VIS_FILE)}")
        print(f"  - Detailed ECLAT Patterns: {self.eclat_patterns_parent_dir}")
        print(f"  - Detailed Anomaly Reports: {self.isolation_forest_dir}")
        
        return all_results_data

# --- Main execution block ---
if __name__ == "__main__":
    try:
        print(f"🔄 Initializing analysis with configuration:")
        print(f"  Excel File: {config.EXCEL_FILE_PATH}")
        print(f"  Sheet Name: {config.SHEET_NAME}")
        print(f"  Min Support for Eclat: {config.MIN_SUPPORT}")
        print(f"  Max Itemset Size for Eclat: {config.MAX_ECLAT_ITEMSET_SIZE}\n")


        analyzer = InstagramECLATAnalyzer(
            excel_file=config.EXCEL_FILE_PATH,
            sheet_name=config.SHEET_NAME,
            min_support=config.MIN_SUPPORT,
            max_eclat_itemset_size=config.MAX_ECLAT_ITEMSET_SIZE
        )
        
        analysis_results = analyzer.run_complete_analysis()
        
        if analysis_results:
            print("\n✅ Analysis pipeline completed successfully!")
        else:
            print("\n⚠️ Analysis pipeline completed with issues or no data.")
            
    except FileNotFoundError:
        print(f"❌ Error: Excel file not found at '{config.EXCEL_FILE_PATH}'")
        print("Please ensure the file exists and the path in 'config.py' is correct.")
    except ImportError as e:
        if 'config' in str(e):
            print("❌ Error: Could not import 'config.py'. Make sure it's in the same directory as 'main_analyzer.py'.")
        else: 
            print(f"❌ Import Error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
