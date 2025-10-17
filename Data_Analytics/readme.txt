# 📚 CodeAlpha Books Analysis

### **Brief One Line Summary**
A complete end-to-end data analytics project that collects, analyzes, and visualizes book data using Python — from web scraping to insight generation.

---

## 🧠 Overview
This project demonstrates a full data analytics workflow divided into three major tasks:
1. **Web Scraping** – extracting book data from the web.
2. **Exploratory Data Analysis (EDA)** – cleaning, analyzing, and identifying key trends.
3. **Data Visualization** – creating interactive visual representations and dashboards for insights.

It showcases the practical use of Python in real-world data projects and can serve as a reference for students and beginners in data analytics.

---

## 💡 Problem Statement
With thousands of books available online, it's challenging to understand trends such as pricing, ratings, and availability without structured analysis.  
The goal of this project is to:
- Collect real-time book data from online sources.
- Perform analytical exploration to discover key trends.
- Visualize and communicate findings effectively through charts and dashboards.

---

## 🧰 Tools and Technologies
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming Language | Python 3.x |
| Data Handling | Pandas, NumPy |
| Web Scraping | BeautifulSoup, Requests |
| Visualization | Matplotlib, Seaborn |
| IDE | PyCharm |
| Version Control | Git, GitHub |

---

## ⚙️ Methods

### **Task 1: Web Scraping**
- Used **BeautifulSoup** and **Requests** to extract book data from a target website.
- Collected fields like title, price, rating, and availability.
- Exported the data into a CSV file `books_data.csv` for analysis.

**File:** `Task-1-Web-Scraping/Web-Scraping.py`  
**Output File:** `books_data.csv`

---

### **Task 2: Exploratory Data Analysis (EDA)**
- Loaded and cleaned the `books_data.csv` dataset.
- Handled missing data, converted data types, and removed duplicates.
- Calculated descriptive statistics and explored relationships between variables.
- Generated plots such as price distributions and rating analysis.

**File:** `Task-2-EDA-Analysis/eda_analysis.py`  


---

### **Task 3: Data Visualization**
- Created data-driven visualizations to highlight insights.
- Designed a summary dashboard (`books_analysis_dashboard.png`) to present findings.
- Focused on identifying pricing patterns, popular categories, and high-rated books.

**File:** `Task-3-Data-Visualization/Data_Visualization.py`  
**Outputs:**  
- `books_analysis_dashboard.png`  
- `price_distribution.png`  
- `price_statistics.png`

---

## 🔍 Key Insights
- Books with mid-range prices (e.g., $30–$40) tend to have higher ratings.  
- Certain genres consistently appear among the top-rated books.  
- There’s a strong correlation between book popularity and pricing.  
- Outlier detection revealed unusually high-priced books that skew average price values.

---

## 📊 Dashboard / Model / Output
The final dashboard visually summarizes:
- Price distribution across books.  
- Rating vs Price comparison.  
- Key metrics and summary statistics.

📁 **Dashboard File:** `Task-3-Data-Visualization/books_analysis_dashboard.png`

---

## 🧩 How to Run This Project?

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Muhammad9845/codealpha_tasks.git
   cd CodeAlpha-Books-Analysis
2. **Create and Activate Virtual Environment**
   ```python -m venv .venv
   ```.venv\Scripts\activate
3. **Install Dependencies**
   ```pip install -r requirements.txt
4. **Run the Tasks**
   Web Scraping:
   ```python Task-1-Web-Scraping/Web-Scraping.py
   EDA Analysis:
   ```python Task-2-EDA-Analysis/eda_analysis.py
   Visualization:
   ```python Task-3-Data-Visualization/Data_Visualization.py
📈 Results & Conclusion

This project successfully demonstrates a complete data analytics pipeline —
from data acquisition to visual storytelling.

It highlights the importance of data preprocessing, visualization clarity, and the ability to derive business insights from raw data.
🚀 Future Work

Implement a live dashboard using Streamlit or Power BI.

Automate data updates with web scraping schedulers (e.g., Cron Jobs).

Expand analysis to include user reviews and sentiment analysis.

Integrate machine learning models for price prediction.
👨‍💻 Author & Contact

Author: Muhammad Ayyaz
Email: ayyazahmad7786@gmail.com
GitHub: https://github.com/Muhammad9845
LinkedIn: www.linkedin.com/in/muhammad-ayyaz-4719b9386


