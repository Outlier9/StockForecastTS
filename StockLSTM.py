import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Input
from sqlalchemy import create_engine


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


# 获取清洗后的数据
cleaned_data = fetch_data_from_mysql()

# LSTM模型训练
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(cleaned_data['close'].values.reshape(-1, 1))

X, y = [], []
time_step = 10
for i in range(len(scaled_data) - time_step - 1):
    X.append(scaled_data[i:(i + time_step), 0])
    y.append(scaled_data[i + time_step, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = Sequential()
model.add(Input(shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, batch_size=32)
# model.save('lstm_stock_model.h5')
# 保存模型为 Keras 格式
model.save('my_model.keras')


