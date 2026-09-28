import subprocess
import sys

# install ollama if it has not been installed yet
try:
    import ollama
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
    import ollama

import os
import pandas as pd
import subprocess

# please set file pathways here
repo_path = r"your file path"
prompt_csv_path = os.path.join(repo_path, "prompts", "prompt file")
response_folder = os.path.join(repo_path, "response file path")
response_csv_path = os.path.join(response_folder, "response file")

os.makedirs(response_folder, exist_ok=True)
df = pd.read_csv(prompt_csv_path)

class BaseLlamaAgent:
    def __init__(self, name, personality, task_description, model='llama3', temperature=0.4, top_k=5):
        self.name = name
        self.personality = personality
        self.task_description = task_description
        self.model = model
        self.options = {
            'temperature': temperature,
            'top_k': top_k
        }

    def make_system_prompt(self, role=None):
        role_text = f" Your role: {role}" if role else ""
        return f"""
        You are {self.name}, a {self.personality} language model agent.{role_text}
        Your task: {self.task_description}
        Use any provided reference materials or definitions to perform your task.
        Be concise (2 sentences max per response), reflective, and justify your reasoning when appropriate.
        After your reasoning, conclude your response in the expected output format if specified.
        """

    def chat(self, input_text, role_description=None):
        messages = [
            {'role': 'system', 'content': self.make_system_prompt(role_description)},
            {'role': 'user', 'content': input_text}
        ]
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options=self.options
        )
        return response['message']['content']

agent = BaseLlamaAgent(
    name="Sage",
    personality="careful and analytical",
    task_description="Provide constructive feedback and evaluation on student writing"
)

responses = []
for i, row in df.iterrows():
    prompt = row['prompt']
    try:
        print(f"Generating response for prompt {i+1}/{len(df)}")
        response = agent.chat(prompt)
        responses.append(response)

df["llama_response"] = responses
df.to_csv(response_csv_path, index=False)
