import http.client
import json

# ==========================================
# API 配置占位符 (Configuration Placeholders)
# ==========================================
API_KEY = "sk-f3db7f1d499d4c5eea813aa64b374847de677a03043c5ecc46b932c7cc01d765"  # 请在此处替换为您的实际 Token
BASE_URL = "gmncode.cn/v1"          # API 服务器地址
MODEL_NAME = "gpt-5.4"          # 使用的模型名称
# ==========================================

conn = http.client.HTTPSConnection(BASE_URL)

payload = json.dumps({
   "model": MODEL_NAME,
   "messages": [
      {
         "role": "user",
         "content": "nihao。"
      }
   ],
   "max_tokens": 1000
})

headers = {
   'Accept': 'application/json',
   'Authorization': f'Bearer {API_KEY}',
   'Content-Type': 'application/json'
}

conn.request("POST", "/v1/chat/completions", payload, headers)
res = conn.getresponse()
data = res.read()

# 输出结果
print(data.decode("utf-8"))