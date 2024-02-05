#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ttest_ind


# In[2]:


# Loading the dataset
data = pd.read_csv(r'E:\Software Applications\brex\OnlineNewsPopularity\OnlineNewsPopularity.csv')

data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


# Handling missing values for numerical columns
data.fillna(data.mean(), inplace=True)


# In[7]:


# Detecting outliers using z-score
from scipy import stats

z_scores = stats.zscore(data.select_dtypes(include='number'))
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
data = data[filtered_entries]


# In[8]:


data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')


# In[10]:


data.isnull().sum()


# In[48]:


data.hist(bins=20, figsize=(55, 50))
plt.show()


# In[12]:


#Time of the Day
# Converting the timedelta to timedelta format
data['timedelta'] = pd.to_timedelta(data['timedelta'], unit='s')

# Extracting hour and minute
data['publication_hour'] = data['timedelta'].dt.components['hours']
data['publication_minute'] = data['timedelta'].dt.components['minutes']


# In[13]:


# User Demographics
# Generating random age between 18 and 65
data['user_age'] = np.random.randint(18, 65, size=len(data))

# Generate random gender (0 for male, 1 for female)
data['user_gender'] = np.random.randint(0, 2, size=len(data))


# In[14]:


# Ad Content
# Calculate the number of words in the article title and content
data['title_word_count'] = data['n_tokens_title']
data['content_word_count'] = data['n_tokens_content']


# In[15]:


# Contextual Information
# Assuming 'data_channel' represents the article category, Concatenate the one-hot encoded columns to the DataFrame
data = pd.concat([data, pd.get_dummies(data[['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world']])], axis=1)

# Drop the original categorical columns
data.drop(['data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world'], axis=1, inplace=True)


# In[16]:


# Social Media Engagement, Calculating engagement ratios
data['shares_per_content_length'] = data['shares'] / data['n_tokens_content']
data['comments_per_content_length'] = data['num_self_hrefs'] / data['n_tokens_content']


# In[17]:


# Content Analysis
# Assuming 'global_sentiment_polarity' represents sentiment analysis score
data['global_sentiment_polarity'] = (data['global_sentiment_polarity'] - data['global_sentiment_polarity'].min()) / (data['global_sentiment_polarity'].max() - data['global_sentiment_polarity'].min())


# In[18]:


# Interaction feature
# interaction feature between shares and content sentiment
data['shares_content_sentiment_interaction'] = data['shares'] * data['global_sentiment_polarity']


# In[20]:


# Check for missing values after feature engineering
print(data.isnull().sum())


# In[26]:


# Identifying categorical columns
categorical_columns = ['user_gender']  # Add other categorical columns as needed

# Drop all non-numeric and irrelevant columns
X.drop(['url'], axis=1, inplace=True)

# Split the data into features (X) and target variable (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional, but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[27]:


# Random Forest model
rf_model = RandomForestRegressor()

# Fit the model on the training data
rf_model.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Sort feature importances in descending order
sorted_indices = feature_importances.argsort()[::-1]


# In[28]:


# Select the top N important features
N = 10   # N taken for top 10
selected_features = X_train.columns[sorted_indices][:N]
print("Selected Features:", selected_features)


# In[29]:


plt.figure(figsize=(10, 6))
plt.bar(range(X_train_scaled.shape[1]), feature_importances[sorted_indices])
plt.xticks(range(X_train_scaled.shape[1]), X_train.columns[sorted_indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importances')
plt.show()


# In[30]:


# Select only the top N important features for training
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Initialize Random Forest model
rf_model_selected = RandomForestRegressor()

# Fit the model on the training data with selected features
rf_model_selected.fit(X_train_selected, y_train)

# Optionally, you can evaluate the model on the training data
train_score = rf_model_selected.score(X_train_selected, y_train)
print("Training R-squared Score:", train_score)


# In[31]:


# Get feature importances of the selected features
selected_feature_importances = rf_model_selected.feature_importances_

# Sort selected feature importances in descending order
selected_sorted_indices = selected_feature_importances.argsort()[::-1]

# Plot selected feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(X_train_selected.shape[1]), selected_feature_importances[selected_sorted_indices])
plt.xticks(range(X_train_selected.shape[1]), selected_features[selected_sorted_indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Selected Feature Importances')
plt.show()


# In[32]:


# Spliting the dataset into control and treatment groups

control_group = data.sample(frac=0.5, random_state=1)  # Assuming 50-50 split
treatment_group = data.drop(control_group.index)


# In[44]:


# Define success metrics: creating the Proxy CTR and CR
def estimate_ctr(data):
    # Estimate CTR using available features (e.g., num_imgs, num_videos, num_hrefs)
    total_features = data['num_imgs'] + data['num_videos'] + data['num_hrefs']
    clicked_features = data['shares']  # Using 'shares' column as a proxy for clicks
    ctr = clicked_features / total_features
    return ctr

def estimate_cr(data):
    # Estimate CR using available features (e.g., shares, comments)
    cr = data['shares'] / data['num_imgs']  # Using 'shares' as a proxy for conversions
    return cr


# In[34]:


# Compute proxy metrics for control group
control_ctr = estimate_ctr(control_group)
control_cr = estimate_cr(control_group)

# Compute proxy metrics for treatment group
treatment_ctr = estimate_ctr(treatment_group)
treatment_cr = estimate_cr(treatment_group)


# In[35]:


# Perform t-test for CTR
ctr_t_stat, ctr_p_value = ttest_ind(control_ctr, treatment_ctr)
if ctr_p_value < 0.05:
    print("Significant difference in estimated CTR between control and treatment groups")
else:
    print("No significant difference in estimated CTR between control and treatment groups")


# In[36]:


# Perform t-test for CR
cr_t_stat, cr_p_value = ttest_ind(control_cr, treatment_cr)
if cr_p_value < 0.05:
    print("Significant difference in estimated CR between control and treatment groups")
else:
    print("No significant difference in estimated CR between control and treatment groups")


# In[38]:


# Assuming 'shares' column as a proxy for treatment condition
data['treatment_condition'] = np.where(data['shares'] > data['shares'].mean(), 'RandomForest', 'baseline')


# In[45]:


# Define treatment conditions and experimental setup
baseline_condition = data[data['treatment_condition'] == 'baseline']
rf_condition = data[data['treatment_condition'] == 'RandomForest']

# Analyze the experimental results using a metric
metric_baseline = baseline_condition['shares']
metric_rf = rf_condition['shares']

# Perform t-test to compare the metric between conditions
t_stat, p_value = ttest_ind(metric_baseline, metric_rf)
print("T-test results for the metric:")
print("T-statistic:", t_stat)
print("P-value:", p_value)


# In[46]:


# Visualize metric distributions
plt.figure(figsize=(10, 6))
sns.histplot(data=baseline_condition, x='shares', color='blue', label='Baseline')
sns.histplot(data=rf_condition, x='shares', color='orange', label='Random Forest')
plt.title('Distribution of Shares')
plt.xlabel('Shares')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




