import os
# 设置环境变量以禁用 oneDNN 优化
# 下边这一行需要写在tensorflow之前
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine



# 数据库连接函数
def connect_to_mysql():
    engine = create_engine('mysql+mysqldb://root:jbyoutlier@localhost/stock_data')
    return engine

# 从MySQL获取数据
def fetch_data_from_mysql():
    engine = connect_to_mysql()
    query = "SELECT * FROM ibm_stock"
    stock_data = pd.read_sql(query, engine)
    return stock_data

# 获取清洗后的数据
cleaned_data = fetch_data_from_mysql()

# 数据预处理
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(cleaned_data['close'].values.reshape(-1, 1))

# 创建训练和测试集
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

# 划分训练集和测试集
train_data = scaled_data[0:training_data_len]
X_train, y_train = [], []
time_step = 10

for i in range(len(train_data) - time_step - 1):
    X_train.append(train_data[i:(i + time_step), 0])
    y_train.append(train_data[i + time_step, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# 预测数据准备
X_test, y_test = [], []
test_data = scaled_data[training_data_len - time_step:]

for i in range(len(test_data) - time_step):
    X_test.append(test_data[i:(i + time_step), 0])
    y_test.append(test_data[i + time_step, 0])

X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 加载模型进行预测
model = load_model('my_model.keras')
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # 反归一化预测值

# 反归一化真实值
y_test = np.array(y_test).reshape(-1, 1)  # 将y_test转换为NumPy数组并重塑
y_test = scaler.inverse_transform(y_test)

# 可视化预测结果
train = cleaned_data[:training_data_len]
valid = cleaned_data[training_data_len:]
valid.loc[:, 'Predictions'] = predictions


# 绘制图像
# 这里如果使用中文需要下载字体，所以决定使用英文
plt.figure(figsize=(14, 5))
plt.title('Stock Price Prediction', fontsize=20)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price', fontsize=14)
plt.plot(train['date'], train['close'], label='Train', color='blue')
plt.plot(valid['date'], valid['close'], label='Actual Price', color='green')
plt.plot(valid['date'], valid['Predictions'], label='Predicted Price', color='red')
plt.legend()  # 添加图例
# 添加保存图像的代码
plt.savefig('stock_price_predictions.png')  # 保存图像
plt.show()


# 计算和输出评估指标
rmse = np.sqrt(np.mean(np.square(predictions - y_test)))
mae = np.mean(np.abs(predictions - y_test))
print(f'RMSE: {rmse}, MAE: {mae}')
