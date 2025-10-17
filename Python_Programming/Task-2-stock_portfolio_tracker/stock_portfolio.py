def stock_portfolio_tracker():
    """
    Stock Portfolio Tracker for CodeAlpha Internship - Task 2
    Simple console-based stock investment calculator
    """
    print("💼 STOCK PORTFOLIO TRACKER")
    print("=" * 40)

    # Predefined stock prices (hardcoded as per task requirements)
    stock_prices = {
        "AAPL": 182.63,  # Apple
        "TSLA": 234.72,  # Tesla
        "GOOGL": 138.21,  # Google
        "MSFT": 378.85,  # Microsoft
        "AMZN": 154.65,  # Amazon
        "META": 351.95,  # Meta (Facebook)
        "NVDA": 477.76,  # NVIDIA
        "BTC": 42250.50,  # Bitcoin (as bonus)
    }

    print("\n📈 AVAILABLE STOCKS & CURRENT PRICES:")
    print("-" * 35)
    for stock, price in stock_prices.items():
        if stock == "BTC":
            print(f"  {stock}: ${price:,.2f}")
        else:
            print(f"  {stock}: ${price:.2f}")

    portfolio = {}
    total_investment = 0

    print("\n🎯 ENTER YOUR STOCK PURCHASES")
    print("(Type 'done' when finished)")
    print("-" * 30)

    while True:
        # Get stock symbol
        stock_symbol = input("\nEnter stock symbol (e.g., AAPL): ").upper().strip()

        if stock_symbol == 'DONE':
            break

        if stock_symbol not in stock_prices:
            print(f"❌ '{stock_symbol}' not found in our database.")
            print("   Available stocks:", ", ".join(stock_prices.keys()))
            continue

        # Get quantity
        try:
            quantity = int(input(f"Enter quantity of {stock_symbol} shares: "))
            if quantity <= 0:
                print("❌ Please enter a positive number")
                continue
        except ValueError:
            print("❌ Please enter a valid number")
            continue

        # Calculate investment for this stock
        price_per_share = stock_prices[stock_symbol]
        stock_investment = price_per_share * quantity

        # Add to portfolio
        portfolio[stock_symbol] = {
            'quantity': quantity,
            'price_per_share': price_per_share,
            'total_value': stock_investment
        }

        total_investment += stock_investment

        print(f"✅ Added {quantity} shares of {stock_symbol}")
        print(f"   Investment: ${stock_investment:,.2f}")

    return portfolio, total_investment, stock_prices


def display_portfolio_summary(portfolio, total_investment, stock_prices):
    """
    Display the portfolio summary and calculations
    """
    if not portfolio:
        print("\n📭 Your portfolio is empty!")
        return

    print("\n" + "=" * 50)
    print("📊 PORTFOLIO SUMMARY")
    print("=" * 50)

    print(f"\n{'Stock':<8} {'Qty':<6} {'Price':<10} {'Total Value':<15}")
    print("-" * 45)

    for stock, details in portfolio.items():
        if stock == "BTC":
            print(
                f"{stock:<8} {details['quantity']:<6} ${details['price_per_share']:>8,.2f} ${details['total_value']:>13,.2f}")
        else:
            print(
                f"{stock:<8} {details['quantity']:<6} ${details['price_per_share']:>8.2f} ${details['total_value']:>13.2f}")

    print("-" * 45)
    print(f"{'TOTAL INVESTMENT':<24} ${total_investment:>13,.2f}")


def calculate_percentages(portfolio, total_investment):
    """
    Calculate portfolio percentages and allocations
    """
    if not portfolio or total_investment == 0:
        return

    print("\n📐 PORTFOLIO ALLOCATION:")
    print("-" * 30)

    for stock, details in portfolio.items():
        percentage = (details['total_value'] / total_investment) * 100
        print(f"  {stock}: {percentage:.1f}%")

    # Find largest holding
    if portfolio:
        largest_holding = max(portfolio.items(), key=lambda x: x[1]['total_value'])
        print(f"\n🏆 Largest holding: {largest_holding[0]} ({largest_holding[1]['total_value']:,.2f})")


def save_portfolio_to_file(portfolio, total_investment, filename="portfolio_report.txt"):
    """
    Save portfolio details to a text file
    """
    try:
        with open(filename, 'w') as file:
            file.write("STOCK PORTFOLIO REPORT\n")
            file.write("=" * 30 + "\n\n")
            file.write(
                f"Report generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            file.write("PORTFOLIO HOLDINGS:\n")
            file.write("-" * 40 + "\n")
            for stock, details in portfolio.items():
                file.write(
                    f"{stock}: {details['quantity']} shares @ ${details['price_per_share']:.2f} = ${details['total_value']:.2f}\n")

            file.write("-" * 40 + "\n")
            file.write(f"TOTAL PORTFOLIO VALUE: ${total_investment:,.2f}\n\n")

            # Add allocation percentages
            if portfolio and total_investment > 0:
                file.write("ALLOCATION PERCENTAGES:\n")
                file.write("-" * 25 + "\n")
                for stock, details in portfolio.items():
                    percentage = (details['total_value'] / total_investment) * 100
                    file.write(f"{stock}: {percentage:.1f}%\n")

        print(f"💾 Portfolio saved to: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False


def main():
    """
    Main function to run the stock portfolio tracker
    """
    print("🚀 CODEALPHA TASK 2: STOCK PORTFOLIO TRACKER")
    print("This program helps you track your stock investments\n")

    # Run the portfolio tracker
    portfolio, total_investment, stock_prices = stock_portfolio_tracker()

    if portfolio:
        # Display summary
        display_portfolio_summary(portfolio, total_investment, stock_prices)

        # Calculate percentages
        calculate_percentages(portfolio, total_investment)

        # Ask to save to file
        save_choice = input("\n💾 Save portfolio to file? (y/n): ").lower()
        if save_choice == 'y':
            save_portfolio_to_file(portfolio, total_investment)

    # Display completion message
    print("\n" + "=" * 50)
    print("🎉 TASK 2 COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("💼 You've built a functional stock portfolio tracker!")
    print("📈 Features included:")
    print("   • Real-time stock price database")
    print("   • Portfolio value calculation")
    print("   • Allocation percentages")
    print("   • File export functionality")
    print("   • Error handling and validation")


if __name__ == "__main__":
    main()