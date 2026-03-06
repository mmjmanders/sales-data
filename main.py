import pandas as pd
import numpy as np


def main():
    # 1. Create a synthetic dataset
    np.random.seed(42)  # Ensuring the "random" data stays consistent for this run
    data = {
        'Product': np.random.choice(['Laptop', 'Tablet', 'Smartphone', 'Smartwatch'], 100),
        'Units_Sold': np.random.randint(1, 50, 100),
        'Price_Per_Unit': np.random.uniform(100, 1200, 100).round(2),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 100)
    }

    df = pd.DataFrame(data)

    # 2. Feature Engineering: Calculate total revenue per row
    df['Total_Revenue'] = df['Units_Sold'] * df['Price_Per_Unit']

    # 3. Data Analysis: Group by Product to see performance
    summary = df.groupby('Product').agg({
        'Units_Sold': 'sum',
        'Total_Revenue': 'sum'
    }).sort_values(by='Total_Revenue', ascending=False)

    # 4. Show the results
    print("--- First 5 Rows of Raw Data ---")
    print(df.head())
    print("\n--- Revenue Summary by Product ---")
    print(summary)

    # 5. Quick Insight: Which region had the highest average sale price?
    avg_region = df.groupby('Region')['Price_Per_Unit'].mean().idxmax()
    print(f"\nInsight: The {avg_region} region has the highest average unit price.")


if __name__ == "__main__":
    main()
