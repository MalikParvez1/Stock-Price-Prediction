import requests

# Konstanten
invest_amount = 10000  # Investitionsbetrag in Euro
api_url = "https://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-17&end=2023-06-29"  # API-URL für historische Bitcoin-Preise


# Funktion zum Abrufen der historischen Bitcoin-Preise
def get_historical_prices():
    response = requests.get(api_url)
    data = response.json()
    return data["bpi"]


# Funktion zum Investieren in Bitcoins
def invest_in_bitcoins():
    prices = get_historical_prices()
    dates = list(prices.keys())
    prices_list = list(prices.values())
    bitcoins = 0
    total_invested = 0
    total_sold = 0

    for i in range(1, len(prices_list)):
        current_price = prices_list[i]
        previous_price = prices_list[i - 1]

        if current_price > previous_price:
            bitcoins_purchased = invest_amount / previous_price
            bitcoins += bitcoins_purchased
            total_invested += invest_amount
            print("{}: Gekauft - {} Bitcoins".format(dates[i], bitcoins_purchased))

        elif current_price < previous_price:
            bitcoins_sold = bitcoins
            bitcoins = 0
            total_sold += bitcoins_sold
            total_invested -= total_sold * current_price
            print("{}: Verkauft - {} Bitcoins".format(dates[i], bitcoins_sold))

    final_balance = bitcoins * prices_list[-1] + total_invested
    total_profit_loss = final_balance - invest_amount
    remaining_balance = final_balance - total_sold * prices_list[-1]
    depot_balance = total_sold * prices_list[-1]

    print("Endgültiger Kontostand: {:.2f} Euro".format(final_balance))
    print("Gesamtgewinn/-verlust: {:.2f} Euro".format(total_profit_loss))
    print("Restguthaben: {:.2f} Euro".format(remaining_balance))
    print("Depotguthaben: {:.2f} Euro".format(depot_balance))


# Hauptprogramm
invest_in_bitcoins()
