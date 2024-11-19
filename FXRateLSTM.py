import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sqlalchemy import create_engine


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

# 获取清洗后的数据
cleaned_data = fetch_data_from_mysql()

# 特征缩放
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(cleaned_data['close'].values.reshape(-1, 1))

# 分割训练和测试集
# 数据前80%用作训练集，后20%作为验证集
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# 创建时间序列数据集函数
def create_dataset(data, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# 准备训练和测试数据集
# 时间窗口，模型基于前20个时间步长进行预测。可以尝试不同长度（如20、30）并观察效果变化。
time_step = 10  # 调整时间窗口
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# 调整形状以适应LSTM输入格式
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 构建LSTM模型
model = Sequential()
model.add(Input(shape=(X_train.shape[1], 1)))
model.add(LSTM(units=64, return_sequences=True))
# 调整Dropout： 当前设置为0.3，可以尝试在不同的层上使用不同的Dropout值，例如0.2~0.5之间。Dropout的作用是防止过拟合。
model.add(Dropout(0.3))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=32))
model.add(Dropout(0.2))
# 增加回归层： 如果希望更高的拟合精度，可以添加多个Dense层，例如在输出前再增加一层Dense。
model.add(Dense(units=1))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 学习率调度器函数
def scheduler(epoch, lr):
    # 每10个epoch将学习率减少一半
    if epoch % 10 == 0 and epoch:
        return lr * 0.5
    return lr

# 定义早停和学习率调度回调
# 提前停止（Early Stopping）： 使用EarlyStopping可以自动停止训练，当模型的验证误差不再改善时，避免过拟合。
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# 学习率调整： 默认adam优化器，但可以试试使用LearningRateScheduler来自定义学习率，例如开始用较高学习率，逐渐减小。
# 使用的学习率。学习率较小通常有助于训练的稳定性，避免大幅度的参数更新，但可能导致训练速度变慢
lr_scheduler = LearningRateScheduler(scheduler)

# 训练模型
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    # 批大小（batch_size）： 尝试不同的batch_size，例如16、32、64，以找到训练稳定性和准确性之间的平衡
    batch_size=32,
    callbacks=[early_stopping, lr_scheduler]
)

# 保存模型为 Keras 格式
model.save('my_model.keras')

# 评估模型性能
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
# 训练集上计算的损失值，数值越小表示模型在训练数据上的拟合效果越好
print(f"Training Loss: {train_loss:.4f}")
# 测试集上计算的损失值，反映了模型在未见过的数据上的表现。测试损失略高于训练损失，但差距不大，说明模型在新数据上的表现依然良好。
print(f"Testing Loss: {test_loss:.4f}")
