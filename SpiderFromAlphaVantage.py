import mysql.connector
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Alpha Vantage API keyAlpha Vantage
API_KEY = 'FYG0G5H1J6BGKPJD'


# 股票数据请求函数
def fetch_stock_data(symbol):
    """
    从Alpha Vantage API获取股票数据。

    参数:
    - symbol (str): 股票代码，"IBM"

    返回:
    - DataFrame: 包含时间序列的DataFrame，带有日期和股票价格及涨跌幅。
    """
    all_data = pd.DataFrame()
    start_date = pd.to_datetime("1999-11-01")  # 转换为Timestamp
    end_date = pd.to_datetime("today")  # 当前日期，转换为Timestamp
    total_requests = 0  # 用于跟踪请求总数

    print(f"开始爬取 {symbol} 的股票数据...")

    # 分批请求数据（因为Alpha Vantage对请求有限制）
    with tqdm(total=(end_date - start_date).days) as pbar:
        while start_date < end_date:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={API_KEY}'

            # 发送请求
            response = requests.get(url)
            data = response.json()

            # 检查是否成功获取数据
            if "Time Series (Daily)" in data:
                # 提取每日数据
                daily_data = data["Time Series (Daily)"]
                if not daily_data:  # 检查daily_data是否为空
                    print("No data found for the specified period. Ending data retrieval.")
                    break

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

                # 筛选出在指定日期范围内的数据
                df = df.loc[start_date:end_date]

                # 检查DataFrame是否为空
                if df.empty:
                    print("No new data available after the start date. Ending data retrieval.")
                    break

                # 计算涨跌幅
                df['Close'] = df['Close'].astype(float)  # 确保 'Close' 列为浮点数
                df['Price Change %'] = df['Close'].pct_change() * 100  # 计算涨跌幅
                df['Price Change %'] = df['Price Change %'].fillna(0)  # 填充第一个值的 NaN

                # 合并数据
                all_data = pd.concat([all_data, df])

                # 更新起始日期为当前数据的最后日期
                start_date = df.index[-1] + pd.Timedelta(days=1)  # 下一次请求从最后日期的后一天开始

                # 更新进度条
                pbar.update(len(df))

                total_requests += 1

                # 限制请求频率，避免触发 API 限制
                time.sleep(15)  # 睡眠15秒
            else:
                print("Error fetching data:", data)
                break

    print(f"完成爬取 {symbol} 的股票数据，共请求了 {total_requests} 次。")
    # 返回合并后的DataFrame
    return all_data


def store_to_mysql(df):
    # 连接到MySQL数据库
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='jbyoutlier',
        database='stock_data'
    )

    cursor = conn.cursor()

    # 创建插入数据的SQL语句
    insert_query = """
    INSERT INTO ibm_stock (date, open, high, low, close, volume, price_change)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        open = VALUES(open),
        high = VALUES(high),
        low = VALUES(low),
        close = VALUES(close),
        volume = VALUES(volume),
        price_change = VALUES(price_change);
    """

    # 循环遍历DataFrame并插入数据
    for index, row in df.iterrows():
        cursor.execute(insert_query,
                       (index.date(), row['Open'], row['High'], row['Low'], row['Close'], int(row['Volume']),
                        row['Price Change %']))

    conn.commit()  # 提交事务
    cursor.close()
    conn.close()


# 调用函数获取数据并打印
stock_data = fetch_stock_data("IBM")
print(stock_data.head())
# 调用函数将数据存储到数据库
if stock_data is not None and not stock_data.empty:
    store_to_mysql(stock_data)

# 绘制收盘价时间序列图
if not stock_data.empty:
    stock_data['Close'] = stock_data['Close'].astype(float)  # 转换数据类型
    stock_data['Close'].plot(title="IBM Daily Close Price", figsize=(12, 6))
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.show()

# 保存为CSV文件
if not stock_data.empty:
    stock_data.to_csv("IBM_stock_data.csv")
