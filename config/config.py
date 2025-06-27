import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

class Keys:
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")