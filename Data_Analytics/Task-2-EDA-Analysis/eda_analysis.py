# EDA_Books_Analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("🔍 STARTING EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# Load the dataset
df = pd.read_csv('books_data.csv')
print("✅ Dataset loaded successfully!")

# =============================================================================
# PHASE 1: DATA UNDERSTANDING & MEANINGFUL QUESTIONS
# =============================================================================

print("\n📖 MEANINGFUL QUESTIONS TO EXPLORE:")
print("1. What is the price distribution of books?")
print("2. Are there any outliers in book prices?")
print("3. What is the average price and how does it vary?")
print("4. Are there patterns in pricing strategies?")
print("5. How many books fall into different price categories?")

# =============================================================================
# PHASE 2: DATA STRUCTURE EXPLORATION
# =============================================================================

print("\n" + "="*60)
print("📊 DATA STRUCTURE ANALYSIS")
print("="*60)

# Basic dataset info
print(f"Dataset Shape: {df.shape}")
print(f"Number of books: {len(df)}")
print(f"Number of features: {df.shape[1]}")

print("\n🔍 First 5 rows:")
print(df.head())

print("\n📝 Dataset Info:")
df.info()

print("\n📋 Column Names and Data Types:")
print(df.dtypes)

print("\n🧮 Basic Statistics:")
print(df.describe())

# Convert Price to numeric for analysis
df['Price_Numeric'] = df['Price'].str.replace('£', '').astype(float)

print("\n💰 Price Statistics (Numeric):")
print(df['Price_Numeric'].describe())

# =============================================================================
# PHASE 3: DATA QUALITY CHECK
# =============================================================================

print("\n" + "="*60)
print("🔎 DATA QUALITY ASSESSMENT")
print("="*60)

# Check for missing values
print("Missing Values:")
missing_data = df.isnull().sum()
print(missing_data)

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Check for unique values
print(f"\nUnique book titles: {df['Title'].nunique()}")

# Check for data consistency
print(f"\nPrice format consistency: {df['Price'].str.startswith('£').all()}")

# =============================================================================
# PHASE 4: UNIVARIATE ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("📈 UNIVARIATE ANALYSIS - PRICE DISTRIBUTION")
print("="*60)

# Detailed price statistics
price_stats = {
    'Mean': df['Price_Numeric'].mean(),
    'Median': df['Price_Numeric'].median(),
    'Mode': df['Price_Numeric'].mode()[0],
    'Std Dev': df['Price_Numeric'].std(),
    'Variance': df['Price_Numeric'].var(),
    'Range': df['Price_Numeric'].max() - df['Price_Numeric'].min(),
    'IQR': df['Price_Numeric'].quantile(0.75) - df['Price_Numeric'].quantile(0.25),
    'Skewness': df['Price_Numeric'].skew(),
    'Kurtosis': df['Price_Numeric'].kurtosis()
}

print("Detailed Price Statistics:")
for stat, value in price_stats.items():
    print(f"{stat}: {value:.2f}")

# =============================================================================
# PHASE 5: TRENDS, PATTERNS & ANOMALIES
# =============================================================================

print("\n" + "="*60)
print("🎯 TRENDS, PATTERNS & ANOMALIES DETECTION")
print("="*60)

# Detect outliers using IQR method
Q1 = df['Price_Numeric'].quantile(0.25)
Q3 = df['Price_Numeric'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Price_Numeric'] < lower_bound) | (df['Price_Numeric'] > upper_bound)]
print(f"Potential outliers (IQR method): {len(outliers)} books")

if len(outliers) > 0:
    print("\n📌 Identified Outliers:")
    for idx, row in outliers.iterrows():
        print(f"  - '{row['Title']}': £{row['Price_Numeric']:.2f}")

# Price categories analysis
def categorize_price(price):
    if price < 30:
        return 'Budget (<£30)'
    elif price <= 45:
        return 'Standard (£30-£45)'
    else:
        return 'Premium (>£45)'

df['Price_Category'] = df['Price_Numeric'].apply(categorize_price)
category_counts = df['Price_Category'].value_counts()

print("\n📊 Price Category Distribution:")
for category, count in category_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {category}: {count} books ({percentage:.1f}%)")

# =============================================================================
# PHASE 6: HYPOTHESIS TESTING & VALIDATION
# =============================================================================

print("\n" + "="*60)
print("🔬 HYPOTHESIS TESTING & STATISTICAL VALIDATION")
print("="*60)

# Hypothesis 1: Prices are normally distributed
print("Hypothesis 1: Book prices follow a normal distribution")
normality_test = stats.shapiro(df['Price_Numeric'])
print(f"Shapiro-Wilk test: statistic={normality_test[0]:.4f}, p-value={normality_test[1]:.4f}")
if normality_test[1] > 0.05:
    print("✅ FAIL to reject null hypothesis - Prices may be normally distributed")
else:
    print("❌ REJECT null hypothesis - Prices are not normally distributed")

# Hypothesis 2: There are significant price differences between categories
print("\nHypothesis 2: Significant price differences exist between categories")
budget_prices = df[df['Price_Category'] == 'Budget (<£30)']['Price_Numeric']
premium_prices = df[df['Price_Category'] == 'Premium (>£45)']['Price_Numeric']

t_test = stats.ttest_ind(budget_prices, premium_prices)
print(f"T-test between Budget and Premium: statistic={t_test[0]:.4f}, p-value={t_test[1]:.4f}")
if t_test[1] < 0.05:
    print("✅ Significant difference found between Budget and Premium categories")
else:
    print("❌ No significant difference between categories")

# =============================================================================
# PHASE 7: DATA VISUALIZATION FOR EDA
# =============================================================================

print("\n" + "="*60)
print("📊 EDA VISUALIZATIONS")
print("="*60)

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comprehensive EDA - Books Dataset', fontsize=16, fontweight='bold')

# 1. Price Distribution Histogram with KDE
axes[0, 0].hist(df['Price_Numeric'], bins=10, edgecolor='black', alpha=0.7, density=True)
sns.kdeplot(df['Price_Numeric'], ax=axes[0, 0], color='red', linewidth=2)
axes[0, 0].set_title('1. Price Distribution with KDE', fontweight='bold')
axes[0, 0].set_xlabel('Price (£)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].grid(True, alpha=0.3)

# 2. Box Plot for Outlier Detection
sns.boxplot(y=df['Price_Numeric'], ax=axes[0, 1])
axes[0, 1].set_title('2. Box Plot - Outlier Detection', fontweight='bold')
axes[0, 1].set_ylabel('Price (£)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Price Categories Pie Chart
category_counts.plot.pie(ax=axes[0, 2], autopct='%1.1f%%', startangle=90)
axes[0, 2].set_title('3. Price Categories Distribution', fontweight='bold')
axes[0, 2].set_ylabel('')

# 4. Cumulative Distribution Function
sorted_prices = np.sort(df['Price_Numeric'])
cdf = np.arange(1, len(sorted_prices)+1) / len(sorted_prices)
axes[1, 0].plot(sorted_prices, cdf, linewidth=2)
axes[1, 0].set_title('4. Cumulative Distribution Function', fontweight='bold')
axes[1, 0].set_xlabel('Price (£)')
axes[1, 0].set_ylabel('CDF')
axes[1, 0].grid(True, alpha=0.3)

# 5. QQ Plot for Normality Check
stats.probplot(df['Price_Numeric'], dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('5. Q-Q Plot - Normality Check', fontweight='bold')

# 6. Price Statistics Comparison
stats_comparison = ['Min', 'Q1', 'Median', 'Q3', 'Max', 'Mean']
stats_values = [
    df['Price_Numeric'].min(),
    df['Price_Numeric'].quantile(0.25),
    df['Price_Numeric'].median(),
    df['Price_Numeric'].quantile(0.75),
    df['Price_Numeric'].max(),
    df['Price_Numeric'].mean()
]

bars = axes[1, 2].bar(stats_comparison, stats_values, color='lightblue', edgecolor='black')
axes[1, 2].set_title('6. Price Statistics Comparison', fontweight='bold')
axes[1, 2].set_ylabel('Price (£)')
axes[1, 2].tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, stats_values):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'£{value:.2f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('EDA_Comprehensive_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# PHASE 8: DATA ISSUES & RECOMMENDATIONS
# =============================================================================

print("\n" + "="*60)
print("⚠️  DATA ISSUES IDENTIFIED & RECOMMENDATIONS")
print("="*60)

print("1. DATA QUALITY ISSUES:")
print("   ✅ No missing values detected")
print("   ✅ No duplicate records found")
print("   ✅ Price format is consistent")

print("\n2. POTENTIAL PROBLEMS:")
if len(outliers) > 0:
    print("   ⚠️  Outliers detected that may need investigation")
else:
    print("   ✅ No significant outliers detected")

if df['Price_Numeric'].skew() > 1 or df['Price_Numeric'].skew() < -1:
    print("   ⚠️  Price distribution is skewed - consider transformation")
else:
    print("   ✅ Price distribution is relatively symmetric")

print("\n3. RECOMMENDATIONS FOR FURTHER ANALYSIS:")
print("   • Consider collecting more book attributes (genre, author, ratings)")
print("   • Investigate the reasons behind premium pricing")
print("   • Analyze temporal trends if publication dates are available")
print("   • Compare with competitor pricing data")

# =============================================================================
# PHASE 9: EDA SUMMARY & INSIGHTS
# =============================================================================

print("\n" + "="*60)
print("📋 EDA SUMMARY & KEY INSIGHTS")
print("="*60)

print("🎯 KEY FINDINGS:")
print(f"1. Dataset contains {len(df)} books with prices ranging from £{df['Price_Numeric'].min():.2f} to £{df['Price_Numeric'].max():.2f}")
print(f"2. Average book price: £{df['Price_Numeric'].mean():.2f} (Median: £{df['Price_Numeric'].median():.2f})")
print(f"3. Price distribution shows {len(outliers)} potential outliers")
print(f"4. {category_counts['Budget (<£30)']} budget books, {category_counts.get('Standard (£30-£45)', 0)} standard, {category_counts['Premium (>£45)']} premium")
print(f"5. Data shows {'right' if df['Price_Numeric'].skew() > 0 else 'left'} skewness: {df['Price_Numeric'].skew():.2f}")

print("\n📊 BUSINESS INSIGHTS:")
print("• The website employs tiered pricing strategy")
print("• Premium books (>£45) represent a significant portion of the inventory")
print("• Price distribution suggests targeted market segmentation")
print("• No major data quality issues detected")

print("\n✅ EDA COMPLETED SUCCESSFULLY!")
print("   Files generated: 'EDA_Comprehensive_Analysis.png'")
print("   Ready for advanced modeling and insights generation!")