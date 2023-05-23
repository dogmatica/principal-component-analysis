#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # **Part I: Research Question**
# 
# ## Research Question
# 
# My data set for this data mining exercise includes data on a telco company’s current and former subscribers, with an emphasis on customer churn (whether customers are maintaining or discontinuing their subscription to service).  Data analysis performed on the dataset will be aimed with this research question in mind: how many principal components does the data set contain when using the continuous numerical data in the data set as input?  Continuous numerical data will include numerical data which includes a measurable variable, rather than numerical data used as a label.

# ---
# 
# ## Objectives and Goals
# 
# Conclusions gleaned from the analysis of this data can benefit stakeholders by revealing information on how far the dimensionality of this data set might be reduced.  Such information may be used to both reduce the feature set and potentially identify variables that exhibit covariance, indicating they may be related.  My goal will be to determine how many principal compenents the continuous data contains, as well as the explained variance of each principal component.

# ---
# 
# # **Part II: Method Justification**
# 
# ## Principal Component Analysis
# 
# Principal Component Analysis, or "PCA", is an unsupervised learning technique.  It utilizes only continuous data and does not take into consideration any target variables.  It is primarily a dimensionality reduction technique.  It uses a covariance matrix to identify highly correlated features and represent those features as a smaller number of uncorrelated features.  The algorithm continues this correlation reduction in an attempt to identify directions of maximum variance in the original data and projecting them onto a reduced dimensional dimensional space.  The resulting components are called "principal components" (Pramoditha, 2020).  
# 
# PCA assumes that there exists a correlation between the features in a data set.  PCA will not be able to determine any principal components in a data set within which no correlation between features is present (Keboola, 2022).
# 
# The expected outcome will be a low number of principal components (which satisfy the Kaiser criterion) to which this original group of continuous variables can be reduced while still maintaining the correlation and variance characteristics of the original data.

# ---
# 
# # **Part III: Data Preparation**
# 
# ## Data Preparation Goals and Data Manipulations
# 
# I would like my data to include only variables relevant to my research question, and to be clean and free of missing values and duplicate rows.  PCA can only operate on continuous variables, so my first goal in data preparation is to make sure the data I will be working with contains no categorical data.
# 
# 
# A list of the variables I will be using for my analysis is included below, along with their variable types and a brief description of each.
# 
# * Population - **continuous** - *Population within a mile radius of customer*
# * Children - **continuous** - *Number of children in customer’s household*
# * Age - **continuous** - *Age of customer*
# * Income - **continuous** - *Annual income of customer*
# * Outage_sec_perweek - **continuous** - *Average number of seconds per week of system outages in the customer’s neighborhood*
# * Email - **continuous** - *Number of emails sent to the customer in the last year*
# * Contacts - **continuous** - *Number of times customer contacted technical support*
# * Yearly_equip_failure - **continuous** - *The number of times customer’s equipment failed and had to be reset/replaced in the past year*
# * Tenure - **continuous** - *Number of months the customer has stayed with the provider*
# * MonthlyCharge - **continuous** - *The amount charged to the customer monthly*
# * Bandwidth_GB_Year - **continuous** - *The average amount of data used, in GB, in a year by the customer*
# * Item1: Timely response - **continuous** - *survey response - scale of 1 to 8 (1 = most important, 8 = least important)*
# * Item2: Timely fixes - **continuous** - *survey response - scale of 1 to 8 (1 = most important, 8 = least important)*
# * Item3: Timely replacements - **continuous** - *survey response - scale of 1 to 8 (1 = most important, 8 = least important)*
# * Item4: Reliability - **continuous** - *survey response - scale of 1 to 8 (1 = most important, 8 = least important)*
# * Item5: Options - **continuous** - *survey response - scale of 1 to 8 (1 = most important, 8 = least important)*
# * Item6: Respectful response - **continuous** - *survey response - scale of 1 to 8 (1 = most important, 8 = least important)*
# * Item7: Courteous exchange - **continuous** - *survey response - scale of 1 to 8 (1 = most important, 8 = least important)*
# * Item8: Evidence of active listening - **continuous** - *survey response - scale of 1 to 8 (1 = most important, 8 = least important)*
# 
# ---
# 
# 
# My first steps will be to import the complete data set and execute functions that will give me information on its size and the data types of its variables.  I will then narrow the data set to a new dataframe containing only the variables I am concerned with, and then utilize functions to determine if any null values or duplicate rows exist.  By using the index_col parameter in my import I utilize CaseOrder, the data set's natural index column, as the index column in my pandas dataframe.

# In[1]:


# Imports and housekeeping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[2]:


# Import the main dataset
df = pd.read_csv('churn_clean.csv', dtype={'locationid':np.int64}, index_col=[0])


# In[3]:


# Display data frame info
df.info()


# In[4]:


# Display data frame top 5 rows
df.head()


# In[5]:


# Trim data frame to variables relevant to research question
columns = ['Population', 'Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email', 'Contacts', 
           'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'Item1', 'Item2', 
           'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8']
df_data = pd.DataFrame(df[columns])
# Store the data frame in variable 'X'
X = df_data


# In[6]:


# Check data for null or missing values
df_data.isna().any()


# In[7]:


# Check data for duplicated rows
df_data.duplicated().sum()


# In[8]:


# Display new data frame top 5 rows
df_data.head()


# ---
# 
# ## Summary Statistics
# 
# I can use the describe() function to display the summary statistics for the entire dataframe, as well as each variable I'll be evaluating for inclusion in the PCA exercise.

# In[9]:


# Display summary statistics for entire data frame
df_data.describe()


# ---
# 
# ## Further Preparation Steps
# 
# I will use the StandardScaler function to scale my variables for more accurate feature weighting.  StandardScaler transforms each variable value to have a mean of 0 and a variance of 1.  Once done, every variable value will fall between -1 and 1, and the data set values can be considered "standardized".  The standardized data set is then assigned to variable "X_scaled".

# In[10]:


# Scaling continuous variables with StandardScaler
scaler = StandardScaler()
scaler.fit(X)
StandardScaler(copy=True, with_mean=True, with_std=True)
X_scaled = scaler.transform(X)


# ---
# 
# ## Copy of Prepared Data Set
# 
# Below is the code used to export the prepared data set to CSV format.

# In[11]:


df_prepared = pd.DataFrame(X_scaled, columns=df_data.columns)
# Export prepared dataframe to csv
df_prepared.to_csv(r'C:\Users\wstul\d212\churn_clean_prepared.csv')


# ---
# 
# # **Part IV: Analysis**
# 
# ## Matrix of All Principal Components
# 
# To begin performing my PCA analysis of the data I instantiated a PCA model using the number of features in the original data set.  The model is then fitted to the scaled data.  The scaled data is then transformed using the PCA model and rendered as numbered principal components (PC1, PC2, etc.).  A loadings matrix is then generated, displaying a weight value for each data set feature in each principal component.

# In[12]:


pca = PCA(n_components = X.shape[1])
pca.fit(X_scaled)


# In[13]:


df_matrix = pd.DataFrame(pca.transform(X_scaled), columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 
                                                            'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 
                                                            'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 
                                                            'PC18', 'PC19'])


# In[14]:


loadings = pd.DataFrame(pca.components_.T, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 
                                                            'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 
                                                            'PC13', 'PC14', 'PC15', 'PC16', 'PC17', 
                                                            'PC18', 'PC19'], index = df_prepared.columns)
loadings


# ---
# 
# ## Kaiser Criterion
# 
# I will use the Kaiser rule to determine which principal components are most important.  The Kaiser rule works by calculating an eigenvalue for each principal component.  An eigenvalue of 1.0 indicates that a principal component is as releveant as an individual variable from the data set, so principal components that rise above the eigenvalue of 1.0 are considered better.
# 
# These eigenvalues can be visualized using a scree plot, making it easy to determine visually how many key principal components were discovered by PCA.

# In[15]:


cov_matrix = np.dot(df_prepared.T, df_prepared) / X.shape[0]


# In[16]:


eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for eigenvector in pca.components_]
eigenvalues


# In[25]:


#The following code constructs the Scree plot
labels = [str(x) for x in range(1, len(eigenvalues)+1)]
 
plt.bar(x=range(1,len(eigenvalues)+1), height=eigenvalues, tick_label=labels)
plt.ylabel('eigenvalues')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()


# ---
# 
# ## Individual and Cumulative Variance
# 
# Based on the scree plot, the first three principal components are the most important.  Using the code below I have printed the explained variance of each of these principal components, as well as their total explained variance.  Explained variance is defined as a statistical measure of how much variation in a dataset can be attributed to each of the principal components generated by PCA (Kumar, 2022).

# In[18]:


total_eigenvalues = sum(eigenvalues)
var_exp = [(i/total_eigenvalues) for i in sorted(eigenvalues, reverse=True)]
print("Explained Variance for PC1: " + str(var_exp[0]))
print("Explained Variance for PC2: " + str(var_exp[1]))
print("Explained Variance for PC3: " + str(var_exp[2]))


# In[19]:


cum_sum_exp = np.cumsum(var_exp)
print("Total Explained Variance for PC1, PC2 and PC3: " + str(cum_sum_exp[2]))


# In[20]:


# Plot the explained variance against cumulative explained variance
#
cum_sum_exp = np.cumsum(var_exp)
plt.bar(range(0,len(var_exp)), var_exp, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_exp)), cum_sum_exp, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# ---
# 
# ## Results of Data Analysis
# 
# PCA determined there are 3 principal components when the 19 continuous variables of the data set were evaluated, answering the original research question "how many principal components does the data set contain when using the continuous numerical data in the data set as input?".  A graphical representation of the variables influencing each of these principal components is included below.

# In[21]:


loadings_df = pd.DataFrame(pca.components_[0:3, :], 
                        columns=df_data.columns)
maxPC = 1.01 * np.max(np.max(np.abs(loadings_df.loc[0:3, :])))
f, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for i, ax in enumerate(axes):
    pc_loadings = loadings_df.loc[i, :]
    colors = ['C0' if l > 0 else 'C1' for l in pc_loadings]
    ax.axhline(color='#888888')
    pc_loadings.plot.bar(ax=ax, color=colors)
    ax.set_ylabel(f'PC{i+1}')
    ax.set_ylim(-maxPC, maxPC)
plt.tight_layout()
plt.show()


# ---
# 
# # **Web Sources**
# 
# https://github.com/StatQuest/pca_demo/blob/master/pca_demo.py
# 
# https://vitalflux.com/pca-explained-variance-concept-python-example/
# 
# https://medium.com/analytics-vidhya/pca-and-how-to-interpret-it-with-python-8aa664f7a69a
# 
# 

# ---
# 
# # **References**
# 
# 
# Keboola.  (2022, April 2).  *A Guide to Principal Component Analysis (PCA) for Machine Learning.*  https://www.keboola.com/blog/pca-machine-learning
# 
# 
# Pramoditha, R.  (2020, August 3).  *Principal Component Analysis (PCA) with Scikit-learn.*  Towards Data Science.  https://towardsdatascience.com/principal-component-analysis-pca-with-scikit-learn-1e84a0c731b0
# 
# 
# Kumar, A.  (2022, August 11).  *PCA Explained Variance Concepts with Python Example.*  Data Analytics.  https://vitalflux.com/pca-explained-variance-concept-python-example/
