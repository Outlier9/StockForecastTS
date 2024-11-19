import requests
import pandas as pd
from datetime import datetime

# 设置API Key和货币对
API_KEY = 'BSOQ1WYAW4M45M0E'
from_currency = "USD"
to_currency = "EUR"


# 请求Alpha Vantage API，获取外汇数据
def fetch_fx_data(from_currency, to_currency):
    url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&outputsize=full&apikey={API_KEY}'
    response = requests.get(url, timeout=30)

    # 如果请求成功，返回JSON数据
    if response.status_code == 200:
        data = response.json()
        if "Time Series FX (Daily)" in data:
            return data["Time Series FX (Daily)"]
        else:
            print("请求的数据格式错误或没有外汇数据。")
            return None
    else:
        print(f"请求失败，状态码：{response.status_code}")
        return None


# 将外汇数据转换为 DataFrame 并输出
def process_data(data):
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close"
    })
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# 测试函数：获取数据并显示最早的日期
def test_fx_data():
    # 获取数据
    data = fetch_fx_data(from_currency, to_currency)
    if data:
        # 处理数据
        df = process_data(data)

        # 显示数据的最早日期
        earliest_date = df.index.min()
        print(f"最早的数据日期：{earliest_date.date()}")

        # 可选：输出数据的前几行
        print(df.head())


if __name__ == "__main__":
    test_fx_data()
