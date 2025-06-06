# Data Preprocessing: Complete Overview

## Why Preprocess Data?

- Raw data is messy, incomplete, inconsistent, and not ready for ML.
- Preprocessing transforms raw data into clean, structured, and machine learning–friendly data.
- “Garbage in, garbage out”: Good models need good data!

---

## Key Steps of Data Preprocessing

---

### 1. Type Casting (Data Type Conversion)

- **What:** Changing data type of columns (e.g., string to integer).
- **Why:** Algorithms need correct data types (e.g., treating IDs as strings).
- **Types:**
  - **Implicit (Automatic):** Python converts automatically (e.g., `4/2 = 2.0`).
  - **Explicit (Manual):** You change type using functions (e.g., `astype()`).
- **Python Example:**
  ```python
  df['EmpID'] = df['EmpID'].astype(str)
  df['Salaries'] = df['Salaries'].astype('int64')
  ```
- **How to identify:** Use `df.dtypes`, `df.info()` to check types. No plot needed.
- **When & Why:** Always check datatypes before EDA/modeling, especially after import or merge.

---

### 2. Duplicate Handling

- **What:** Identifying and removing repeated rows/columns.
- **Why:** Duplicates can bias statistics and slow down analysis.
- **Detecting Duplicates:**
  ```python
  df.duplicated()             # Boolean series
  sum(df.duplicated())        # Count duplicates
  ```
- **How to identify (Plot):** Use `df.value_counts()` for a column to see frequency, or bar plot for categorical columns. Not usually visualized, but a bar plot may reveal suspicious spikes.
- **Removing Duplicates:**
  ```python
  df = df.drop_duplicates()
  ```
- **Column Duplicates:** Use correlation (see heatmap) to spot highly similar columns (e.g., drop one if corr > 0.85).
- **Scenario:** Always remove full-duplicate rows; drop columns only if one is redundant for prediction.

---

### 3. Outlier Treatment

- **What:** Handling extreme values far from most data.
- **Why:** Outliers can skew results and mislead models.
- **Detection:** Boxplots, IQR method, Z-score.
- **How to identify (Plot):** Use **boxplot** for each numeric column, and **histogram** for distribution. Z-score scatter plot can also help.
- **Treatment Techniques:**

  - **Remove:** Drop outlier rows.
  - **Replace (Cap):** Set outliers to nearest normal value.
  - **Winsorization:** Capping outliers at percentile or statistical limit.
  - **Rectify/Retain:** Only if valid or important outliers.

| Method        | When to Use                            | Python Example                                         | Formula (if any) | Real-World Impact                 |
| ------------- | -------------------------------------- | ------------------------------------------------------ | ---------------- | --------------------------------- |
| Remove        | Outlier is error/typo, not real        | `df = df[(df['col'] >= lower) & (df['col'] <= upper)]` | N/A              | Cleaner statistics, robust models |
| Replace (Cap) | Outlier is real, but effect limited    | see Winsorization                                      | see below        | Avoids bias from outliers         |
| Winsorization | Keep all rows, limit outlier influence | see below                                              | see below        | Preserves dataset size            |
| Retain        | Outlier is valid and important         | N/A                                                    | N/A              | Detects rare, important cases     |

- **Formulas:**
  - **IQR Method:**
    ```
    IQR = Q3 - Q1
    Lower Limit = Q1 - 1.5 * IQR
    Upper Limit = Q3 + 1.5 * IQR
    ```
  - **Z-Score:**
    ```
    z = (x - mu) / sigma
    ```
- **Winsorization in Python:**
  ```python
  from feature_engine.outliers import Winsorizer
  winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['Salaries'])
  df = winsor.fit_transform(df)
  ```
- **Gaussian (Z-Score) Capping:**
  ```python
  winsor = Winsorizer(capping_method='gaussian', tail='both', fold=3, variables=['Salaries'])
  df = winsor.fit_transform(df)
  ```
- **When & Why:**
  - Winsorization: Use when you want to keep all rows but reduce outlier influence (good for regression).
  - Removal: Use if outlier is error/noise.
  - Retain: If outlier is a valid rare event (fraud, spikes).

---

### 4. Zero Variance & Near-Zero Variance

- **What:** Columns with all values the same (or nearly so).
- **Why:** These features carry no predictive power and slow modeling.
- **Python Example:**
  ```python
  zero_var_cols = df.columns[df.var() == 0]
  df = df.drop(columns=zero_var_cols)
  ```
- **How to identify (Plot):** Use **bar plot** or **histogram** —you’ll see a single bar/constant value.
- **Tip:** Set a threshold for "near-zero" (e.g., var < 0.01). Always check after encoding.

---

### 5. Encoding Categorical Variables

- **What:** Convert non-numeric (categorical) data to numbers for ML.
- **Why:** Most algorithms require numerical input.
- **Types:**
  - **Nominal:** No order (e.g., City, Gender)
    - **One-hot encoding (dummy):** New binary column for each value.
      ```python
      pd.get_dummies(df, columns=['City'])
      ```
    - **OneHotEncoder (sklearn):**
      ```python
      from sklearn.preprocessing import OneHotEncoder
      enc = OneHotEncoder()
      enc.fit_transform(df[['City']])
      ```
    - **Best Use:** Nominal, low/medium cardinality (e.g., state, gender).
    - **How to identify (Plot):** Use **countplot/barplot** for each categorical column to visualize class frequencies.
  - **Ordinal:** Has logical order (e.g., Low < Medium < High)
    - **Label Encoding:**
      ```python
      from sklearn.preprocessing import LabelEncoder
      le = LabelEncoder()
      df['Grade'] = le.fit_transform(df['Grade'])
      ```
    - **Ordinal Encoding:**
      ```python
      from sklearn.preprocessing import OrdinalEncoder
      oe = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
      df['Grade'] = oe.fit_transform(df[['Grade']])
      ```
    - **Best Use:** Ordinal, explicit order (education level, risk rating).
    - **How to identify (Plot):** **Boxplot** of target by each ordinal feature can show relationship with target.
- **Best Practice:**
  - Use one-hot for nominal, label/ordinal for ordinal.
  - Avoid one-hot if cardinality > 20 (try target or frequency encoding).
- **Formula:** For one-hot, X_category = 1 if row in category else 0.

---

### 6. Discretization (Binning)

- **What:** Convert continuous variable to discrete bins/categories.
- **Why:** Can improve model stability, reveal hidden patterns, simplify interpretation.
- **Types:**
  - **Fixed-width binning:** Each bin has equal range.
    ```python
    pd.cut(df['Salaries'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])
    ```
  - **Quantile/Adaptive binning:** Each bin has equal number of points.
    ```python
    pd.qcut(df['Salaries'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    ```
  - **Custom bins:** Set exact cutoffs as needed (business logic).
    ```python
    pd.cut(df['age'], bins=[0, 20, 40, 60, 100], labels=['Teen', 'Adult', 'Senior', 'Elder'])
    ```

| Type              | When to Use                | Pros                                   | Cons                      |
| ----------------- | -------------------------- | -------------------------------------- | ------------------------- |
| Fixed-width (cut) | Uniformly distributed data | Simple, interpretable                  | Unbalanced bins if skewed |
| Quantile (qcut)   | Skewed data                | Equal sample sizes, robust to outliers | Uneven ranges             |
| Custom            | Domain-specific needs      | Tailored to business logic             | Needs SME knowledge       |

- **How to identify (Plot):** Use **histogram** (before and after binning) to visualize bins. **Countplot** for each bin to see distribution.
- **Scenario:** Used in credit scoring, age bands, salary segmentation.

---

### 7. Missing Value Treatment

- **What:** Deal with empty (NaN, None) values in data.
- **Why:** Models can’t train with missing values.
- **Types of Missingness:**
  - **MCAR:** Completely random
  - **MAR:** Related to other observed variables
  - **MNAR:** Not at random
- **Techniques:**
  - **Deletion:** Remove missing rows/columns (if few).
    ```python
    df = df.dropna()
    ```
  - **Imputation:** Fill with mean, median, mode, constant, random sample.
    ```python
    from sklearn.impute import SimpleImputer
    # Numeric columns
    df['Salaries'] = SimpleImputer(strategy='median').fit_transform(df[['Salaries']])
    # Categorical columns
    df['Gender'] = SimpleImputer(strategy='most_frequent').fit_transform(df[['Gender']])
    # Constant fill
    df['City'] = SimpleImputer(strategy='constant', fill_value='Unknown').fit_transform(df[['City']])
    ```
  - **Advanced:** Predictive (KNN, regression), random, or library tools like AutoClean.

| Method           | When to Use                     | Pros                   | Cons               |
| ---------------- | ------------------------------- | ---------------------- | ------------------ |
| Deletion         | Small % missing, not important  | Simple, fast           | Wastes data        |
| Mean/Median/Mode | Numeric/categorical, random     | Easy, preserves size   | Can add bias       |
| Constant/Random  | Flags special value/keeps dist. | Handles edge cases     | Not always logical |
| Advanced ML      | Not at random, large missing    | Best preserves pattern | Complex, slow      |

- **How to identify (Plot):** Use **heatmap** (`sns.heatmap(df.isnull())`) to visualize missingness, or barplot of missing value counts per column.
- **Scenario:** Mean/median for numeric, mode for category, ML for large/mixed missingness.

---

### 8. Transformation

- **What:** Change data distribution for normalization or variance stabilization.
- **Why:** Many ML models work best with normal distributions (linear, logistic regression, kNN, SVM, PCA).
- **Types:**

  - **Function Transformation:** log, sqrt, reciprocal, etc.

    - **Log:** Use for right-skewed positive data.
      ```
      x' = log(x) or x' = log(x+1) if zeros present
      ```
    - **Python:**
      ```python
      df['LogSalary'] = np.log1p(df['Salaries'])
      ```
    - **How to identify (Plot):** **Histogram** or **QQ-plot** before and after transformation to check normality.

  - **Power Transformation:**

    - **Box-Cox:** Only positive values.

      ```
      x' = [(x^λ)-1]/λ  if λ ≠ 0,   log(x) if λ = 0
      ```

      ```python
      from scipy.stats import boxcox
      df['Salaries_boxcox'], lam = boxcox(df['Salaries'] + 1)
      ```

    - **Yeo-Johnson:** Works for zero and negative values.

      ```
      x' = see table for formulas (works for x >= 0 and x < 0)
      ```

      ```python
      from scipy.stats import yeojohnson
      df['Salaries_yeo'], lam = yeojohnson(df['Salaries'])
      ```

    - **How to identify (Plot):** **QQ-plot** (`scipy.stats.probplot`) for normality check before/after.

  - **Quantile Transformation:** Make distribution uniform or normal using quantiles.

    ```python
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(output_distribution='normal')
    df['Salary_qt'] = qt.fit_transform(df[['Salaries']])
    ```

    - **How to identify (Plot):** Histogram before/after, QQ-plot for checking target distribution.

| Transformation | When to Use             | Pros                            | Cons                       |
| -------------- | ----------------------- | ------------------------------- | -------------------------- |
| Log            | Positive skew, no zeros | Simple, fast, effective         | Not for zeros/negatives    |
| Box-Cox        | Positive values         | Powerful, flexible              | No zeros/negatives allowed |
| Yeo-Johnson    | Any data                | Handles negatives/zeros         | Slightly more complex      |
| Quantile       | Any distribution        | Uniform/normalizes distribution | May lose original order    |

- **When & Why:**
  - Use Box-Cox for positive data, Yeo-Johnson for mixed data, Log for positive skew.
  - Helps models that are sensitive to distribution shape.

---

### 9. Scaling

- **What:** Put all features on the same scale so one doesn’t dominate.
- **Why:** Improves performance of distance-based and gradient-based models.
- **Types:**
  - **Normalization (Min-Max Scaling):**
    - Scale range to [0, 1].
    - Formula:
      ```
      x' = (x - x_min) / (x_max - x_min)
      ```
    - Python:
      ```python
      from sklearn.preprocessing import MinMaxScaler
      df_scaled = MinMaxScaler().fit_transform(df)
      ```
    - **How to identify (Plot):** **Histogram** (distribution) and **boxplot** (spread) before and after scaling.
    - **When:** Deep learning, image data, when all features need to be strictly bounded.
  - **Standardization (Z-score Scaling):**
    - Center at mean=0, std=1.
    - Formula:
      ```
      z = (x - mu) / sigma
      ```
    - Python:
      ```python
      from sklearn.preprocessing import StandardScaler
      df_z = StandardScaler().fit_transform(df)
      ```
    - **How to identify (Plot):** Histogram and boxplot before/after scaling; should see mean shift to 0, std to 1.
    - **When:** Classic ML (regression, SVM, PCA).
  - **Robust Scaling:**
    - Median=0, IQR=1, robust to outliers.
    - Formula:
      ```
      x' = (x - median) / IQR
      ```
    - Python:
      ```python
      from sklearn.preprocessing import RobustScaler
      df_robust = RobustScaler().fit_transform(df)
      ```
    - **How to identify (Plot):** Histogram and boxplot to see if outliers are less influential after scaling.
    - **When:** When outliers exist but shouldn’t dominate (e.g., income).

| Scaling Method | When to Use               | Pros                        | Cons                   |
| -------------- | ------------------------- | --------------------------- | ---------------------- |
| Min-Max        | NN, images, strict bounds | Bounded, simple             | Sensitive to outliers  |
| Z-score        | Most classic ML           | Standardizes for most algos | Not robust to outliers |
| Robust         | Outliers present          | Handles outliers well       | May not scale tightly  |

---

## Extra Preprocessing Concepts

- **Masking:** Small outlier hidden by a big one.
- **Swamping:** Non-outlier flagged as outlier (often in percent trimming).
- **Correlation-based Redundancy Removal:** Drop one of highly correlated columns (e.g., corr > 0.85). **How to identify (Plot):** **Heatmap** (`sns.heatmap(df.corr())`)
- **Pipeline:** Use `sklearn.pipeline` to automate sequence of preprocessing steps.

---

## Best Practices

- Always explore and visualize before and after each step (e.g., with histograms, boxplots, barplots, heatmaps).
- Document all changes for reproducibility.
- Apply the **same steps to training and test data** for consistency.

---

## Real-World Impact on Model Building

- **Type Casting:** Prevents algorithm errors, speeds up computations.
- **Duplicate Handling:** Removes bias and speeds up analysis.
- **Outlier Treatment:** Prevents extreme values from skewing predictions; especially critical for regression, clustering.
- **Zero Variance:** Removes useless features, faster model fitting.
- **Encoding:** Converts text categories to usable numbers; enables all ML models.
- **Discretization:** Helps with tree-based models, reveals group effects, good for segmentation.
- **Missing Value Treatment:** Keeps as much data as possible, avoids dropped samples, maintains patterns.
- **Transformation:** Makes data fit model assumptions (improves accuracy).
- **Scaling:** Ensures fair contribution from all features; critical for kNN, SVM, PCA, neural networks.

---

_Prepared by Abhijeet Panda_
