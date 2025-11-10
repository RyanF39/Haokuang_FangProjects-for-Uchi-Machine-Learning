#%% Dataset Preparation - Install UCI ML Repo Package
#!pip install ucimlrepo

#%% Dataset Preparation
from ucimlrepo import fetch_ucirepo

# fetch dataset 
online_news_popularity = fetch_ucirepo(id=332) 
  
# data (as pandas dataframes) 
X = online_news_popularity.data.features 
y = online_news_popularity.data.targets 
  
# metadata 
print(online_news_popularity.metadata) 
  
# variable information 
print(online_news_popularity.variables) 

#%% Verifying Project Requirements
import pandas as pd

# 1. Check Row Requirement (Rows: >= 10,000)
num_rows = X.shape[0]
meets_row_req = num_rows >= 10000
print(f"Rows: {num_rows} (Requirement >= 10,000: {meets_row_req})")
 
# 2. Check Feature Requirement (Features: >= 50)
num_features = X.shape[1]
meets_feature_req = num_features >= 50
print(f"Features: {num_features} (Requirement >= 50: {meets_feature_req})")

# 3. Check for Missing Values (for 'data handled properly') 
missing_values = X.isnull().sum().sum()
print(f"Total Missing Values in X: {missing_values}")
 
# 4. Check for Categorical Data (for 'categorical encoding') 
# We check the 'dtype' of each column in the feature dataframe X
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) == 0:
    print("Categorical Columns to Encode: 0")
    print("Note: No categorical encoding is needed for the 'X' features.")
else:
    print(f"Categorical Columns to Encode: {len(categorical_cols)}")
    print(f"Columns: {list(categorical_cols)}")

#%%
# This will show us the *actual* column name in the y DataFrame
print("Column(s) in the 'y' target DataFrame:")
print(y.columns)

# %% EDA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# FIX: Strip whitespace from column names
# This removes leading/trailing spaces, e.g., ' shares' becomes 'shares'
X.columns = X.columns.str.strip()
y.columns = y.columns.str.strip()

# Combine X and y for easier EDA
df = pd.concat([X, y], axis=1)
# Analyse target variable distribution
# Set up a plotting style
sns.set(style="whitegrid")

# 1. Get descriptive statistics for 'shares'
print(df['shares'].describe())

# 2. Plot a histogram to see the distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['shares'], bins=100, kde=True)
plt.title('Distribution of Article Shares')
plt.xlabel('Number of Shares')
plt.ylabel('Frequency')
plt.show()

# 3. Check the distribution again, but with a limit
plt.figure(figsize=(12, 6))
sns.histplot(df[df['shares'] < 5000]['shares'], bins=100, kde=True)
plt.title('Distribution of Article Shares (Zoomed In < 5000 Shares)')
plt.xlabel('Number of Shares')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# From the first plot (Distribution of Article Shares), 
# we know that almost all your data is clustered on the far left.
# Thus, the prediction model will meet a problem: 
# If I feed this to a standard linear regression, 
# the model will be obsessed with trying to correctly predict those few "viral" articles (the 800,000+ ones) 
# and will do a very poor job of predicting the "normal" articles (the 1,000-3,000 share ones), which are the vast majority.
# The second plot ("Zoomed In") shows the distribution of the "normal" articles is still right-skewed. 
# To silve all of this, we must log-transform the target variable. 
# This will compress the huge values and spread out the low values, resulting in a much more "normal" (bell-shaped) distribution that models can learn from.

# %% Log-Transform the Target Variable
df['log_shares'] = np.log1p(df['shares'])

# --- Plot the new distribution ---
print("\n--- Descriptive Statistics for 'log_shares' ---")
print(df['log_shares'].describe())

plt.figure(figsize=(12, 6))
sns.histplot(df['log_shares'], bins=50, kde=True)
plt.title('Distribution of Log-Transformed Shares (log_shares)')
plt.xlabel('Log(1 + Number of Shares)')
plt.ylabel('Frequency')
plt.show()


# %%
# 1. Check feature scales
# .describe() on all features will be too much. Let's look at a sample.
print(df[['n_tokens_content', 'title_subjectivity', 'global_sentiment_polarity']].describe())

# 2. Check for multicollinearity (features correlated with each other)
# This is computationally intensive. Let's just check a subset of 10-15 features.
feature_subset = [
    'n_tokens_title', 'n_tokens_content', 'n_unique_tokens', 
    'n_non_stop_words', 'num_hrefs', 'num_self_hrefs', 
    'average_token_length', 'num_keywords', 'kw_avg_avg',
    'global_subjectivity', 'global_sentiment_polarity', 'title_subjectivity'
]

# Calculate the correlation matrix for the subset
corr_matrix = df[feature_subset].corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature-Feature Correlation Heatmap (Subset)')
plt.show()

# %% [markdown]
# n_tokens_content has a max of 8474, while title_subjectivity has a max of 1.0. Because of this massive difference in scales, 
# you must use a scaler (like StandardScaler from scikit-learn) before training any linear models.
# From the correlation heatmap, we can see that some features are highly correlated. 
# The correlation between n_non_stop_words and n_unique_tokens is 1.00.
# This means they provide redundant information. Keeping both will confuse your linear models.

# %% Feature-Target Relationships
# Which features are most strongly correlated with log_shares?
# Calculate the correlation of all features with our *new* target, 'log_shares'
correlations = df.corr(numeric_only=True)['log_shares'].sort_values()

# Display the top 10 most positively correlated features
print("--- Top 10 Positive Correlations ---")
print(correlations.tail(11)[:-1]) # tail(11) to include the top 10, then [:-1] to exclude 'shares' itself

print("\n--- Top 10 Negative Correlations ---")
# Display the top 10 most negatively correlated features
print(correlations.head(10))

# %% [markdown]
# The analysis of feature-target relationships reveals that an article's topic and keyword popularity are the most significant drivers of its shares. 
# The strongest positive predictor is kw_avg_avg (0.22 correlation), suggesting that articles about already-popular topics are shared more. 
# Publishing on the weekend (is_weekend, 0.11) and belonging to a specific topic (LDA_03, 0.13) are also associated with higher engagement. 
# Conversely, the strongest negative predictors are related to content category, with articles from "Topic 2" (LDA_02, -0.17) and the "World News" channel (data_channel_is_world, -0.15) seeing significantly fewer shares. 
# These findings indicate that what an article is about and when it is published are key factors in predicting its popularity.

# %% Classification Modelings: Logit Regression, SVM, Rondom Forest, KNN.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score 

# Create the Binary Classification Target 
# Set the threshold to 1400, as used in the reference paper
threshold_shares = 1400

print(f"Using a fixed threshold of {threshold_shares} shares.")

# Create the new target column 'is_popular'
# if shares > threshold, 0 otherwise
df['is_popular'] = (df['shares'] > threshold_shares).astype(int)

# Check the balance of our new target
print("\nNew Target Variable 'is_popular':")
print(df['is_popular'].value_counts())
# %% Model Preparation: Train-Test Split
# 1. Data: Define X (features) and y (target)
# X: Drop the redundant feature and all target-related columns
X = df.drop(columns=['shares', 'log_shares', 'n_non_stop_words', 'is_popular']) 
# y: Our new binary target
y = df['is_popular']

# 2. Split Data into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# %% Define Models and Preprocessing Pipeline

# Define the Preprocessing Step
scaler = StandardScaler()

# Define the Models to Compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_jobs=-1, random_state=42),
    "Support Vector Machine (SVM)": SVC() 
}


#%% Train, Predict, and Evaluate Each Model
import time 
from sklearn.metrics import classification_report, accuracy_score

print("\n--- Starting Model Training ---")

results_reports = {} 
model_metrics_list = [] 

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    start_time = time.time()
    
    # Create the full pipeline
    pipeline = Pipeline(steps=[('scaler', scaler),
                              ('model', model)])
    
    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    # 1. Get the string report (for printing later)
    report_string = classification_report(y_test, y_pred)
    results_reports[name] = report_string
    
    # 2. Get the dictionary report (for the table)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # 3. Extract the metrics we want (using 'weighted avg' for a balanced view)
    accuracy = report_dict['accuracy']
    precision = report_dict['weighted avg']['precision']
    recall = report_dict['weighted avg']['recall']
    f1_score = report_dict['weighted avg']['f1-score']
    
    # 4. Append the metrics to our list
    model_metrics_list.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision (Weighted)': precision,
        'Recall (Weighted)': recall,
        'F1-Score (Weighted)': f1_score,
        'Time (s)': time_taken
    })
    
    # This print statement is still useful
    print(f"Done. Accuracy: {accuracy:.4f}")
    print(f"Training & Prediction took: {time_taken:.2f} seconds")

print("\n--- Model Training Complete ---")


for name, report in results_reports.items():
    print(f"\n--- Classification Report for {name} ---")
    print(report)

# Create Final Comparison Table with All Metrics
results_df = pd.DataFrame(model_metrics_list)
results_df = results_df.set_index('Model').sort_values(by="Accuracy", ascending=False)

print("\n--- Final Model Comparison (All Metrics) ---")
# Set display format for floats
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print(results_df)

# %% Feature Importance from Random Forest
# From above, we get the best model is Random Forest.
# 1. Define the RF model again
rf_model = RandomForestClassifier(n_jobs=-1, random_state=42)

# 2. Create the pipeline
rf_pipeline = Pipeline(steps=[('scaler', scaler),
                             ('model', rf_model)])

# 3. Fit it on the *full* training set
print("--- Fitting Random Forest to get feature importances ---")
rf_pipeline.fit(X_train, y_train)

# 4. Get the feature importances from the 'model' step
importances = rf_pipeline.named_steps['model'].feature_importances_

# 5. Get the feature names
# We use the column names from X_train
feature_names = X_train.columns

# 6. Create a DataFrame for easy viewing
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# --- Display the results ---
print("\n--- Top 15 Most Important Features (from Random Forest) ---")
print(importance_df.head(15))

# --- Plot the results ---
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top 15 Most Important Features for Predicting Popularity')
plt.savefig('feature_importance.png')
plt.show()

# %%
