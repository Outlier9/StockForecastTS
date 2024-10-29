import mysql.connector
from sqlalchemy import create_engine
import pandas as pd


# 数据库连接函数
def connect_to_mysql():
    # 创建SQLAlchemy引擎
    engine = create_engine('mysql+mysqldb://root:jbyoutlier@localhost/stock_data')
    return engine

# 从MySQL获取数据
def fetch_data_from_mysql():
    engine = connect_to_mysql()
    query = "SELECT * FROM ibm_stock"
    stock_data = pd.read_sql(query, engine)  # 使用SQLAlchemy引擎
    return stock_data


# 数据清洗函数
def clean_data(stock_data):
    print("检查缺失值：")
    print(stock_data.isnull().sum())

    initial_shape = stock_data.shape  # 记录初始数据形状
    stock_data.dropna(inplace=True)
    print(f"删除缺失值后，数据行数从 {initial_shape[0]} 变为 {stock_data.shape[0]}")

    # 使用正确的列名（小写）
    stock_data['open'] = stock_data['open'].astype(float)
    stock_data['high'] = stock_data['high'].astype(float)
    stock_data['low'] = stock_data['low'].astype(float)
    stock_data['close'] = stock_data['close'].astype(float)
    stock_data['volume'] = stock_data['volume'].astype(int)

    # 继续进行差分和特征提取
    stock_data['close_diff'] = stock_data['close'].diff()
    stock_data.dropna(inplace=True)  # 删除因差分而产生的NaN值

    stock_data['MA20'] = stock_data['close'].rolling(window=20).mean()
    stock_data['MA50'] = stock_data['close'].rolling(window=50).mean()

    def compute_rsi(data, window=14):
        delta = data['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    stock_data['RSI'] = compute_rsi(stock_data)

    print("数据清洗和特征提取完成。")

    return stock_data


# 存储到新表的函数
def store_to_new_table(df):
    conn = connect_to_mysql()
    cursor = conn.cursor()

    # 创建新表的SQL命令
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ibm_stock_cleaned (
        date DATE PRIMARY KEY,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume INT,
        close_diff FLOAT,
        ma20 FLOAT,
        ma50 FLOAT,
        rsi FLOAT
    );
    """)

    # 插入数据
    insert_query = """
    INSERT INTO ibm_stock_cleaned (date, open, high, low, close, volume, close_diff, ma20, ma50, rsi)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    for index, row in df.iterrows():
        cursor.execute(insert_query, (
            index.date(), row['Open'], row['High'], row['Low'], row['Close'],
            int(row['Volume']), row['Close_diff'], row['MA20'], row['MA50'], row['RSI']
        ))

    conn.commit()
    cursor.close()
    conn.close()


# 主程序
if __name__ == "__main__":
    # 从数据库获取数据
    stock_data = fetch_data_from_mysql()
    print("从数据库获取的列名：")
    print(stock_data.columns)
    # 清洗数据
    cleaned_data = clean_data(stock_data)

    # 比较原始数据和清洗后的数据
    if cleaned_data.equals(stock_data):
        print("数据清洗后与原始数据相同，不需要存储到新表。")
    else:
        print("数据发生变化，存储到新表中。")
        store_to_new_table(cleaned_data)
