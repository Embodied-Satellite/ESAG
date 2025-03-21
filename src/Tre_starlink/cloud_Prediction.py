import requests
import json


#https://www.tomorrow.io/
#https://openweathermap.org/


# 填入你在 OpenWeatherMap 上获取的 API 密钥
API_KEY = 'your_api_key'
# 定义要查询的经纬度
LATITUDE = 30.0
LONGITUDE = 120.0
# 构建 API 请求的 URL
URL = f'https://api.openweathermap.org/data/2.5/forecast?lat={LATITUDE}&lon={LONGITUDE}&appid={API_KEY}&units=metric'

try:
    # 发送 HTTP 请求
    response = requests.get(URL)
    # 检查响应状态码
    if response.status_code == 200:
        # 解析响应的 JSON 数据
        data = response.json()
        # 打印前 24 小时（约 8 个时间点，每 3 小时一次预报）的云量预报
        for i in range(min(8, len(data['list']))):
            forecast = data['list'][i]
            # 获取预报的时间戳
            timestamp = forecast['dt_txt']
            # 获取云量信息
            cloud_cover = forecast['clouds']['all']
            print(f"时间: {timestamp}, 云量: {cloud_cover}%")
    else:
        print(f"请求失败，状态码: {response.status_code}")
except requests.RequestException as e:
    print(f"请求发生错误: {e}")
except (KeyError, json.JSONDecodeError) as e:
    print(f"解析数据时出错: {e}")