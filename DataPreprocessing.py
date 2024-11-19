from sqlalchemy import create_engine
import pandas as pd

# 数据库连接函数
def connect_to_mysql():
    # 创建SQLAlchemy引擎
    engine = create_engine('mysql+mysqldb://root:jbyoutlier@localhost/forex_data')
    return engine

# 从MySQL获取数据
def fetch_data_from_mysql():
    engine = connect_to_mysql()
    query = "SELECT * FROM forex_rates ORDER BY date"
    rates_data = pd.read_sql(query, engine)  # 使用SQLAlchemy引擎
    return rates_data

# 数据清洗函数
def clean_data(rates_data):
    print("检查缺失值：")
    print(rates_data.isnull().sum())

    initial_shape = rates_data.shape  # 记录初始数据形状
    rates_data.dropna(inplace=True)
    print(f"删除缺失值后，数据行数从 {initial_shape[0]} 变为 {rates_data.shape[0]}")

    # 使用正确的列名（小写）
    rates_data['open'] = rates_data['open'].astype(float)
    rates_data['high'] = rates_data['high'].astype(float)
    rates_data['low'] = rates_data['low'].astype(float)
    rates_data['close'] = rates_data['close'].astype(float)

    # 处理 Price Change %
    rates_data['price_change'] = rates_data['price_change'].astype(float)  # 确保 'Price Change %' 列为浮点数
    rates_data['price_change'] = rates_data['price_change'].fillna(0)  # 填充缺失值

    # 继续进行差分和特征提取
    rates_data['close_diff'] = rates_data['close'].diff()
    rates_data.dropna(inplace=True)  # 删除因差分而产生的NaN值

    rates_data['MA20'] = rates_data['close'].rolling(window=20).mean()
    rates_data['MA50'] = rates_data['close'].rolling(window=50).mean()

    def compute_rsi(data, window=14):
        delta = data['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    rates_data['RSI'] = compute_rsi(rates_data)

    print("数据清洗和特征提取完成。")

    return rates_data

# 存储到新表的函数
def rates_to_new_table(df):
    conn = connect_to_mysql()
    cursor = conn.cursor()

    # 创建新表的SQL命令
    cursor.execute(""" 
    CREATE TABLE IF NOT EXISTS forex_data_cleaned (
        date DATE PRIMARY KEY,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        price_change FLOAT
    );
    """)

    # 插入数据
    insert_query = """
    INSERT INTO forex_data_cleaned (date, open, high, low, close, price_change)
    VALUES (%s, %s, %s, %s, %s, %s);
    """

    for index, row in df.iterrows():
        cursor.execute(insert_query, (
            index.date(), row['Open'], row['High'], row['Low'], row['Close'],
            row['price_change']
        ))

    conn.commit()
    cursor.close()
    conn.close()

# 主程序
if __name__ == "__main__":
    # 从数据库获取数据
    rates_data = fetch_data_from_mysql()
    print("从数据库获取的列名：")
    print(rates_data.columns)
    # 清洗数据
    cleaned_data = clean_data(rates_data)

    # 比较原始数据和清洗后的数据
    if cleaned_data.equals(rates_data):
        print("数据清洗后与原始数据相同，不需要存储到新表。")
    else:
        print("数据发生变化，存储到新表中。")
        rates_to_new_table(cleaned_data)
