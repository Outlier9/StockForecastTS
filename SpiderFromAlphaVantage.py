# Alpha Vantage API密匙：FYG0G5H1J6BGKPJD
import mysql.connector
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Alpha Vantage API key（替换为您的实际API Key）
API_KEY = 'FYG0G5H1J6BGKPJD'


# 股票数据请求函数
def fetch_stock_data(symbol, function="TIME_SERIES_DAILY", outputsize="compact"):
    """
    从Alpha Vantage API获取股票数据。

    参数:
    - symbol (str): 股票代码，例如 "IBM"
    - function (str): 请求数据类型，默认为每日时间序列
    - outputsize (str): 数据量，"compact"返回最近100天数据，"full"返回全部数据

    返回:
    - DataFrame: 包含时间序列的DataFrame，带有日期和股票价格。
    """
    url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&outputsize={outputsize}&apikey={API_KEY}'

    # 发送请求
    response = requests.get(url)
    data = response.json()

    # 检查是否成功获取数据
    if "Time Series (Daily)" in data:
        # 提取每日数据
        daily_data = data["Time Series (Daily)"]
        # 转换为DataFrame
        df = pd.DataFrame.from_dict(daily_data, orient="index")
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
            "5. volume": "Volume"
        })
        # 将日期转换为日期格式，并按日期排序
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df
    else:
        print("Error fetching data:", data)
        return None


def store_to_mysql(df):
    # 连接到MySQL数据库
    conn = mysql.connector.connect(
        host='localhost',
        user='root',  # 替换为你的数据库用户名
        password='jbyoutlier',  # 替换为你的数据库密码
        database='stock_data'  # 替换为你创建的数据库名
    )

    cursor = conn.cursor()

    # 创建插入数据的SQL语句
    insert_query = """
    INSERT INTO ibm_stock (date, open, high, low, close, volume)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        open = VALUES(open),
        high = VALUES(high),
        low = VALUES(low),
        close = VALUES(close),
        volume = VALUES(volume);
    """

    # 循环遍历DataFrame并插入数据
    for index, row in df.iterrows():
        cursor.execute(insert_query,
                       (index.date(), row['Open'], row['High'], row['Low'], row['Close'], int(row['Volume'])))

    conn.commit()  # 提交事务
    cursor.close()
    conn.close()




# 调用函数获取数据并打印
stock_data = fetch_stock_data("IBM")
print(stock_data.head())
# 调用函数将数据存储到数据库
if stock_data is not None:
    store_to_mysql(stock_data)

# 绘制收盘价时间序列图
stock_data['Close'] = stock_data['Close'].astype(float)  # 转换数据类型
stock_data['Close'].plot(title="IBM Daily Close Price", figsize=(12, 6))
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

# stock_data.to_csv("IBM_stock_data.csv")