import os
import pandas as pd
from openai import OpenAI
from time import sleep
from datetime import datetime

# Set your uploaded and save file path
repo_path = r"Your file path"
prompt_csv_path = os.path.join(repo_path, "prompt file") 
response_folder = os.path.join(repo_path, "feedback file path") 
response_csv_path = os.path.join(response_folder, "feedback file")
temp_csv_path = os.path.join(response_folder, "temp_progress.csv") #If there is a large amount of data, it is recommended to store it during the process
os.makedirs(response_folder, exist_ok=True)

# ⚙️ configure setting
API_KEY = "Please set your OpenAI API Key" 
MODEL_NAME = "gpt-4o-mini" 
# or MODEL_NAME = "gpt-5-mini" 
MAX_RETRIES = 3
RETRY_DELAY = 5
REQUEST_INTERVAL = 0.5 
BATCH_SAVE_INTERVAL = 20
MAX_TOKENS = 1024

try:
    df = pd.read_csv(prompt_csv_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(prompt_csv_path, encoding='latin1')

df = df.reset_index(drop=True)

if os.path.exists(temp_csv_path):
    temp_df = pd.read_csv(temp_csv_path, encoding='utf-8-sig')
    responses = temp_df["gpt4omini_response"].tolist()
    start_idx = len(responses)

    
else:
    responses = []
    start_idx = 0


client = OpenAI(api_key=API_KEY)

def query_gpt4omini(prompt):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()


for i in range(start_idx, len(df)):
    prompt = df.at[i, "prompt"]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}]  [{i+1}/{len(df)}] Prompt: {prompt[:80]}...")

    response = query_gpt4omini(prompt)
    responses.append(response)
    print(f"Response: {response[:100]}...")

    if (i + 1) % BATCH_SAVE_INTERVAL == 0 or i == len(df) - 1:
        df_temp = df.iloc[:i+1].copy()
        df_temp["gpt4omini_response"] = responses
        df_temp.to_csv(temp_csv_path, index=False, encoding='utf-8-sig')
        print(f"save the process（{i+1}/{len(df)}）：{temp_csv_path}")

    sleep(REQUEST_INTERVAL)

df["gpt4omini_response"] = responses
df.to_csv(response_csv_path, index=False, encoding='utf-8-sig')
