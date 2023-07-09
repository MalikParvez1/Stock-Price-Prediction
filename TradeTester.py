import random
from sklearn.preprocessing import MinMaxScaler as sc
import time

def trade_test(y_test, y_pred):
    # Simulating trades
    virtual_account_balance = 10000  # Initial virtual account balance in USD
    holding_coins = 0  # Number of coins held
    buy_threshold = 0.00005  # Threshold for buying (2% increase in price)
    sell_threshold = -0.00005  # Threshold for selling (2% decrease in price)
    total_profit_loss = 0  # Track total profit/loss


    for i in range(len(y_pred)):
        predicted_price = y_pred[i]
        actual_price = y_test[i]

        if predicted_price >= actual_price * (1 + buy_threshold):
            # Buy signal
            if holding_coins == 0:
                coins_to_buy = virtual_account_balance / actual_price
                holding_coins = coins_to_buy
                virtual_account_balance = 0
                print(f"Buy {coins_to_buy} coins at price {actual_price}")

        elif predicted_price <= actual_price * (1 + sell_threshold):
            # Sell signal
            if holding_coins > 0:
                coins_to_sell = holding_coins
                virtual_account_balance = coins_to_sell * actual_price
                holding_coins = 0
                print(f"Sell {coins_to_sell} coins at price {actual_price}")
                # Calculate profit/loss
                profit_loss = (virtual_account_balance - 10000) / 10000
                total_profit_loss += profit_loss

        # Print current balance and holdings
        print(f"Virtual Account Balance: {virtual_account_balance} USD")
        print(f"Holding Coins: {holding_coins} coins")

    # Calculate percentage return on investment
    percentage_return = (virtual_account_balance - 10000) / 10000 * 100

    print("Total Profit/Loss:", total_profit_loss)
    print("Percentage Return on Investment:", percentage_return, "%")

    return percentage_return


#trades randomly 50/50 probability for buying / selling
def trade_fifty_fifty(y_test, y_pred):
    # Simulating trades
    virtual_account_balance = 10000  # Initial virtual account balance in USD
    holding_coins = 0  # Number of coins held
    total_profit_loss = 0  # Track total profit/loss

    for i in range(len(y_pred)):
        predicted_price = y_pred[i]
        actual_price = y_test[i]

        # Generate random number between 1 and 2
        random_number = random.randint(1, 2)

        if random_number == 1:
            # Buy signal
            if holding_coins == 0:
                coins_to_buy = virtual_account_balance / actual_price
                holding_coins = coins_to_buy
                virtual_account_balance = 0
                print(f"Buy {coins_to_buy} coins at price {actual_price}")

        elif random_number == 2:
            # Sell signal
            if holding_coins > 0:
                coins_to_sell = holding_coins
                virtual_account_balance = coins_to_sell * actual_price
                holding_coins = 0
                print(f"Sell {coins_to_sell} coins at price {actual_price}")
                # Calculate profit/loss
                profit_loss = (virtual_account_balance - 10000) / 10000
                total_profit_loss += profit_loss

        # Print current balance and holdings
        print(f"Virtual Account Balance: {virtual_account_balance} USD")
        print(f"Holding Coins: {holding_coins} coins")

    # Calculate percentage return on investment
    percentage_return = (virtual_account_balance - 10000) / 10000 * 100

    print("Total Profit/Loss:", total_profit_loss)
    print("Percentage Return on Investment:", percentage_return, "%")
    return percentage_return


def calculate_hodling_return(y_test):
    # Get the first and last scaled values of y_test
    first_value = y_test[0][0]
    last_value = y_test[-1][0]

    # Calculate the percentage return on investment
    percentage_return = (last_value - first_value) / first_value * 100

    # Print the percentage return on investment
    print("Percentage Return on Investment:", percentage_return, "%")

    return percentage_return
