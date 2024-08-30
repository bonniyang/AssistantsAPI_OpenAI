from openai import OpenAI
import os
from dotenv import load_dotenv
from assistantfunction import *
import numpy as np

load_dotenv()

api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)

passage = "Your passage"

def params_tuning(prompt: str, temperature: float,  top_p: float, assistant_name: str, vectorstore_name: str):

    assistant = get_assistant_if_exist(assistant_name)
    vectorstore = get_vector_store_if_exist(vectorstore_name)

    if assistant is None:
        print(f"assistant '{assistant_name}' not found.")
    if vectorstore is None:
        print(f"vector store '{vectorstore_name}' not found.")

    thread = client.beta.threads.create(
        messages=[{"role": "user", "content": prompt}]
        , tool_resources={
            "file_search": {
                "vector_store_ids": [vectorstore.id]

            }
        }
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
        temperature=temperature,
        top_p=top_p,
    )

    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text
    message_value = message_content.value
    value = re.sub(r'【\d+:\d+†source】', '', message_value)
    return value

assistant_name= 'new assistant'
vectorstore_name= 'new vs'
temperature_values = np.arange(0.2,1.0,0.3)
top_p_values = np.arange(0.2,1.0,0.3)

for temp in temperature_values:
    for top_p in top_p_values:
        result = params_tuning(passage, temp, top_p, assistant_name, vectorstore_name)
        print(f"Temperature: {temp}, Top_p: {top_p}, Result: {result}")


