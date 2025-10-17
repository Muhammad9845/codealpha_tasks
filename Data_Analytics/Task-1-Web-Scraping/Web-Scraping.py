import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Set up Chrome driver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

try:
    # Open the website
    url = "http://books.toscrape.com"
    driver.get(url)
    time.sleep(2)  # Wait for page to load

    # Collect data
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    books_data = []
    # Find all book containers on the page
    for book in soup.find_all('article', class_='product_pod'):
        # Extract book title
        title = book.h3.a['title']
        # Extract book price
        price = book.find('p', class_='price_color').text

        # Add to our data list
        books_data.append({
            'Title': title,
            'Price': price
        })

    # Create DataFrame
    df = pd.DataFrame(books_data)
    print(df.head())  # Display first 5 rows
    df.to_csv('books_selenium.csv', index=False)
    print("✅ Data saved successfully!")

finally:
    # Close the browser
    driver.quit()

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    print("✅ All libraries working in PyCharm!")
    print("Numpy version:", np.__version__)
    print("Pandas version:", pd.__version__)
