import mysql.connector
from sqlalchemy import create_engine
import requests
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


# Alpha Vantage API key
API_KEY = 'FYG0G5H1J6BGKPJD'
# API_KEY = 'R1S1PUBU3VIUPWMJ'
# API_KEY = 'W7PY4H2JPYHTUUU8'
# API_KEY = 'A0IF78SH2FMWHEYE'
# API_KEY = 'NR71RVIBFFT00KBA'
# API_KEY = 'DGD2E9M4XB2YPK2J'
# API_KEY = 'P0OMG676MRV1HNDR'
# API_KEY = 'QKHFQ7BBOC2MG67W'
# API_KEY = 'BSOQ1WYAW4M45M0E'


def store_to_mysql(df, from_currency, to_currency):
    import mysql.connector

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="jbyoutlier",
        database="forex_data"
    )
    cursor = conn.cursor()

    saved_rows = 0
    for index, row in df.iterrows():
        date = index.strftime("%Y-%m-%d")
        open_price = row["Open"]
        high = row["High"]
        low = row["Low"]
        close = row["Close"]
        price_change = row["Price Change %"]

        # 插入或更新数据
        insert_query = """
            INSERT INTO forex_rates (date, open, high, low, close, price_change)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                price_change = VALUES(price_change);
        """
        cursor.execute(insert_query, (date, open_price, high, low, close, price_change))
        saved_rows += 1

    conn.commit()
    cursor.close()
    conn.close()

    return saved_rows


# 外汇汇率数据请求函数
def fetch_forex_data(from_currency, to_currency):
    start_date = pd.to_datetime("2014-11-16")
    end_date = pd.to_datetime("2024-11-18")
    total_saved_rows = 0
    all_data = pd.DataFrame()

    with tqdm(total=(end_date - start_date).days, desc="爬取进度") as pbar:
        while start_date <= end_date:  # 改为 <= 确保终止日期包含在内
            next_date = start_date + pd.Timedelta(days=365)
            if next_date > end_date:
                next_date = end_date

            print(f"请求日期范围：{start_date.date()} 至 {next_date.date()}")  # 添加调试信息

            url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_currency}&to_symbol={to_currency}&outputsize=full&apikey={API_KEY}'
            response = requests.get(url, timeout=30)
            data = response.json()

            if "Time Series FX (Daily)" in data:
                daily_data = data["Time Series FX (Daily)"]
                if not daily_data:
                    print("未找到数据，结束爬取。")
                    break

                df = pd.DataFrame.from_dict(daily_data, orient="index")
                df = df.rename(columns={
                    "1. open": "Open",
                    "2. high": "High",
                    "3. low": "Low",
                    "4. close": "Close"
                })
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df = df.loc[start_date:next_date]

                df['Close'] = df['Close'].astype(float)
                df['Price Change %'] = df['Close'].pct_change() * 100
                df['Price Change %'] = df['Price Change %'].fillna(0)

                # 保存数据到数据库
                saved_rows = store_to_mysql(df, from_currency, to_currency)
                total_saved_rows += saved_rows
                all_data = pd.concat([all_data, df])

                if not df.empty:
                    start_date = df.index[-1] + pd.Timedelta(days=1)
                else:
                    start_date = next_date  # 如果没有数据，直接跳到下一周期

                pbar.update(len(df))
                print(f"本次保存数据 {saved_rows} 条，数据库中已存入 {total_saved_rows} 条。")
                print("———————————————分界线———————————————")


                time.sleep(15)
            else:
                print("请求数据时出错:", data)
                break
    time.sleep(3)
    print(f"完成爬取 {from_currency}-{to_currency} 的外汇汇率数据，共保存 {total_saved_rows} 条数据。")
    return all_data


def fetch_all_data():
    """从数据库中读取所有存储的数据"""
    # 使用 SQLAlchemy 创建连接引擎
    engine = create_engine(
        "mysql+mysqlconnector://root:jbyoutlier@localhost/forex_data"
    )
    query = "SELECT * FROM forex_rates ORDER BY date"

    # 使用 pandas 读取 SQL 数据
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    df['date'] = pd.to_datetime(df['date'])  # 确保日期列为 datetime 类型
    return df


def plot_forex_data(df):
    """绘制外汇汇率折线图"""
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['close'], label="USD to EUR", color="blue")
    plt.title("USD to EUR Exchange Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate (Close Price)")
    plt.legend()
    plt.grid()
    # 保存图像
    plt.savefig('forex_exchange_rate.png', dpi=300, bbox_inches='tight')
    plt.show()



def export_to_csv(df, filename="USD_EUR_forex_data.csv"):
    """导出数据为CSV文件"""
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"数据已成功导出到 {filename}")


# 主程序
if __name__ == "__main__":
    # 调用爬虫爬取数据
    print("开始爬取 USD-EUR 的外汇汇率数据...")
    try:
        fetch_forex_data("USD", "EUR")
    except Exception as e:
        print(f"爬取过程中出现错误：{e}")

    # 从数据库中读取所有数据
    print("从数据库中读取所有数据...")
    all_data = fetch_all_data()

    # 绘制图表
    print("开始绘制图表...")
    plot_forex_data(all_data)

    # 导出为 CSV 文件
    print("导出数据为CSV文件...")
    export_to_csv(all_data)
