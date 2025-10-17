import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("🎨 Starting Task 3: Data Visualization")
print("=" * 50)

# Load and prepare data
df = pd.read_csv('books_data.csv')
df['Price_Numeric'] = df['Price'].str.replace('£', '').astype(float)

# Calculate statistics needed for the script
avg_price = df['Price_Numeric'].mean()
median_price = df['Price_Numeric'].median()
most_expensive = df.loc[df['Price_Numeric'].idxmax()]
least_expensive = df.loc[df['Price_Numeric'].idxmin()]

# Set up professional styling
plt.style.use('default')
sns.set_palette("husl")

print("📊 Creating Comprehensive Visualizations...")

# 1. MAIN DASHBOARD - 4 charts in one figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Books Data Analysis Dashboard - CodeAlpha Internship', fontsize=16, fontweight='bold')

# Chart 1: Price Distribution Histogram
axes[0, 0].hist(df['Price_Numeric'], bins=8, edgecolor='black', alpha=0.7, color='lightblue')
axes[0, 0].set_title('1. Distribution of Book Prices', fontweight='bold', pad=20)
axes[0, 0].set_xlabel('Price (£)')
axes[0, 0].set_ylabel('Number of Books')
axes[0, 0].grid(True, alpha=0.3)

# Add statistics lines
axes[0, 0].axvline(avg_price, color='red', linestyle='--', linewidth=2, label=f'Average: £{avg_price:.2f}')
axes[0, 0].axvline(median_price, color='green', linestyle='--', linewidth=2, label=f'Median: £{median_price:.2f}')
axes[0, 0].legend()

# Chart 2: Box Plot for Price Spread
axes[0, 1].boxplot(df['Price_Numeric'], vert=True, patch_artist=True)
axes[0, 1].set_title('2. Book Prices Spread Analysis', fontweight='bold', pad=20)
axes[0, 1].set_ylabel('Price (£)')
axes[0, 1].set_xticklabels(['All Books'])
axes[0, 1].grid(True, alpha=0.3)

# Chart 3: Top 10 Most Expensive Books
top_10 = df.nlargest(10, 'Price_Numeric')
bars = axes[1, 0].barh(range(len(top_10)), top_10['Price_Numeric'], color='orange', alpha=0.7)
axes[1, 0].set_title('3. Top 10 Most Expensive Books', fontweight='bold', pad=20)
axes[1, 0].set_xlabel('Price (£)')
axes[1, 0].set_yticks(range(len(top_10)))
axes[1, 0].set_yticklabels(top_10['Title'], fontsize=9)
axes[1, 0].invert_yaxis()

# Add price labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    axes[1, 0].text(width + 1, bar.get_y() + bar.get_height()/2, f'£{width:.2f}',
                   ha='left', va='center', fontsize=8)

# Chart 4: Price Categories Pie Chart
price_ranges = ['Budget (<£30)', 'Standard (£30-£45)', 'Premium (>£45)']
budget = len(df[df['Price_Numeric'] < 30])
standard = len(df[(df['Price_Numeric'] >= 30) & (df['Price_Numeric'] <= 45)])
premium = len(df[df['Price_Numeric'] > 45])
sizes = [budget, standard, premium]
colors = ['lightgreen', 'lightblue', 'gold']

wedges, texts, autotexts = axes[1, 1].pie(sizes, labels=price_ranges, colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 10})
axes[1, 1].set_title('4. Books by Price Category', fontweight='bold', pad=20)

# Enhance pie chart text
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig('books_analysis_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Dashboard saved as 'books_analysis_dashboard.png'")

# 2. ADDITIONAL INDIVIDUAL VISUALIZATIONS

# Price Trend Chart
plt.figure(figsize=(14, 6))
plt.plot(range(len(df)), df['Price_Numeric'], marker='o', linewidth=1, markersize=4, alpha=0.7)
plt.title('Book Prices Distribution', fontweight='bold', pad=20)
plt.xlabel('Book Index')
plt.ylabel('Price (£)')
plt.grid(True, alpha=0.3)
plt.axhline(y=avg_price, color='r', linestyle='--', label=f'Average: £{avg_price:.2f}')
plt.legend()
plt.savefig('price_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical Summary Chart
plt.figure(figsize=(12, 6))
stats_data = {
    'Min': df['Price_Numeric'].min(),
    'Q1': df['Price_Numeric'].quantile(0.25),
    'Median': df['Price_Numeric'].median(),
    'Q3': df['Price_Numeric'].quantile(0.75),
    'Max': df['Price_Numeric'].max(),
    'Average': df['Price_Numeric'].mean()
}

colors = ['lightcoral', 'lightyellow', 'lightblue', 'lightgreen', 'lightpink', 'orange']
bars = plt.bar(stats_data.keys(), stats_data.values(), color=colors, edgecolor='black')

plt.title('Price Statistics Summary', fontweight='bold', pad=20)
plt.ylabel('Price (£)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, stats_data.values()):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'£{value:.2f}',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('price_statistics.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Additional visualizations created successfully!")

# 3. DATA STORYTELLING SUMMARY
print("\n" + "="*60)
print("📖 DATA STORY SUMMARY")
print("="*60)
print(f"• The dataset contains {len(df)} books with prices ranging from £{df['Price_Numeric'].min():.2f} to £{df['Price_Numeric'].max():.2f}")
print(f"• Average book price: £{avg_price:.2f}")
print(f"• Price distribution shows {budget} budget books, {standard} standard, and {premium} premium books")
print(f"• Most expensive book: '{most_expensive['Title']}' at £{most_expensive['Price_Numeric']:.2f}")
print(f"• Least expensive book: '{least_expensive['Title']}' at £{least_expensive['Price_Numeric']:.2f}")
print(f"• The data reveals varied pricing strategy across different book titles")

print("\n🎯 TASK 3 COMPLETED: Data Visualization Successful!")