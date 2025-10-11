
import csv
import math
import sys

def analyze_prices(filename):
    prices = []
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header
        for i, row in enumerate(reader):
            try:
                if len(row) == 4:
                    price = float(row[3])
                    prices.append(price)
                else:
                    pass
            except (ValueError, IndexError) as e:
                continue

    if not prices:
        print("No valid prices found.")
        return

    min_price = min(prices)
    max_price = max(prices)
    mean_price = sum(prices) / len(prices)
    if len(prices) > 1:
        variance = sum([(p - mean_price) ** 2 for p in prices]) / (len(prices) - 1)
        std_dev = math.sqrt(variance)
    else:
        std_dev = 0

    print(f"Number of valid prices: {len(prices)}")
    print(f"Min price: {min_price}")
    print(f"Max price: {max_price}")
    print(f"Mean price: {mean_price}")
    print(f"Standard deviation of price: {std_dev}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_prices(sys.argv[1])
    else:
        print("Please provide the CSV file path as an argument.")
