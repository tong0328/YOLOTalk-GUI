import lineTool
import requests

# Change 'token_key' to your Line token
def line_notify(msg):
    token_key = ''   # 測試 
    header = {'Content-Type':'application/x-www-form-urlencoded',"Authorization":'Bearer '+token_key}
    URL = 'https://notify-api.line.me/api/notify'
    payload = {'message':msg}
    res=requests.post(URL,headers=header,data=payload)
