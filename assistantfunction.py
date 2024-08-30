from typing import Optional, List

from openai import OpenAI
from openai.types.beta.vector_store import VectorStore
from openai.types.beta.vector_stores import VectorStoreFileBatch
from openai.types.beta.assistant import Assistant
import re
import os
from dotenv import load_dotenv

from project_path import PROJECT_ROOT_DIR

load_dotenv()

api_key = os.getenv("API_KEY")
client = OpenAI(api_key=api_key)


# create assistant and vector store by name

def get_vector_store_if_exist(vs_name: str) -> VectorStore:
    """

    :param vs_name:
    :return:
    Usage:
    >>> get_vector_store_if_exist('testing_not_found')
    Out[25]: VectorStore(id='vs_TkyY9I2QfpK1Ir ...

    >>> get_vector_store_if_exist('blahblah')
    """
    result_vs = None
    vector_stores = client.beta.vector_stores.list()
    for vectorstore in vector_stores:
        if vectorstore.name == vs_name:
            result_vs = vectorstore
            break
    return result_vs


def get_assistant_if_exist(assist_name: str) -> Assistant:
    """
    :param assist_name: name of the assistant
    Usage:
    # assistant not found
    >>> get_assistant_if_exist('assitant_not_found')
    # assistant found
    >>> get_assistant_if_exist('assistant_found')
    Out[30]: Assistant(id='asst_UeSMzgFeZ ...
    """
    result_assist = None
    assist_list = client.beta.assistants.list()
    for assist in assist_list:
        if assist.name == assist_name:
            result_assist = assist
            break
    return result_assist


def get_assistant(assistant_name: str,
                  vectorstore_name: Optional[str] = None) -> Assistant:
    """
    get an assistant from openai and assign a vector store with the assistant
    :param assistant_name: name of the assistant
    :param vectorstore_name: name of the vector store
    :return:

    Usage:
    >>> assistant_name = 'assistant testing'
    >>> vectorstore_name = 'testing_vs'

    # vector store not found
    >>> vectorstore_name = 'testing_not_found'

    # get an assistant where it's already existed
    >>> assistant_name = 'assistant_testing'
    >>> get_assistant(assistant_name)
    Out[30]: Assistant(id='asst_UeSMzgFeZS5AL15yY ...

    >>> get_assistant(assistant_name, 'testing_vs')
    Out[30]: Assistant(id='asst_UeSMzgFeZS5AL15y ...

    # create new assistant but using existing vs
    >>> assistant_name = 'bonny_new'
    >>> vectorstore_name = 'testing_vs'
    >>> get_assistant(assistant_name, vectorstore_name)
    Out[30]: Assistant(id='asst_UeSMzgFeZS5AL1 ...

    # create new assistant and assign a new vector store
    >>> assistant_name = 'bonny_new2'
    >>> vectorstore_name = 'testing_not_found'
    >>> get_assistant(assistant_name, vectorstore_name)
    Out[30]: Assistant(id='asst_UeSMzgFeZS5AL1 ...
    """
    assist_obj = get_assistant_if_exist(assistant_name)
    if assist_obj is None:
        if vectorstore_name is not None:
            vs_obj = get_vector_store_if_exist(vectorstore_name)
            if vs_obj is None:
                vs_obj = client.beta.vector_stores.create(name=vectorstore_name)
            resource_attr = {
                "file_search": {
                    "vector_store_ids": [vs_obj.id]
                }
            }
        else:
            resource_attr = None

        assist_obj = client.beta.assistants.create(
            name=assistant_name,
            instructions= 'put your instruction',
            model="gpt-4o",
            tools=[{"type": "file_search"}],
            tool_resources=resource_attr,
        )
    return assist_obj


# create function to upload file to vector store
def upload_file(file_path: List[str], vectorstore_name: str) -> VectorStoreFileBatch:
    """

    :param file_path:
    :param vectorstore_name:
    :return:
    Usage:
    >>> file_path = ['/Users/bonniyang/resources/file1.pdf']
    >>> vectorstore_name = ''
    """
    vs = get_vector_store_if_exist(vectorstore_name)
    assert vs is not None, 'vector store name not found [{}]'.format(vectorstore_name)
    file_streams = [open(path, "rb") for path in file_path]
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vs.id, files=file_streams
    )
    return file_batch


def ask_assistant(
        question: str,
        assistant_name: str, vectorstore_name: str) -> str:

    """
    ask an existing assistant a question and use the vector store as thread's tool resource
    if assistant does not exist, will create a new assistant using the vector store
    :param question: the question to be asked
    :param assistant_name: name of the assistant
    :param vectorstore_name: name of the vector store
    :return: the response

    Usage:
    >>> question = 'Your question'
    >>> assistant_name = 'assistant testing'
    >>> vectorstore_name = 'vs testing'
    """
    assistant = get_assistant(assistant_name,vectorstore_name)
    vs_obj = get_vector_store_if_exist(vectorstore_name)
    return ask_assistant_with_id(question, assistant_id=assistant.id, vectorstore_id=vs_obj.id)

def ask_assistant_with_id(question: str,
                          assistant_id: str,
                          vectorstore_id: str
                          ) -> str:
    """
    ask an existing assistant a question using assistnat id and use the vector store id as thread's tool resource
    :param question: the question to be asked
    :param assistant_id: id of the assistant
    :param vectorstore_id: id of the vector store
    :return: the response

    Usage:
    >>> question = 'Your question'
    >>> assistant_id = 'asst_ekrjnegkegeg'
    >>> vectorstore_id = 'vs_rkokrengegmeg'
    """
    thread = client.beta.threads.create(
        messages=[{"role": "user", "content": question}]
        , tool_resources={
            "file_search": {
                "vector_store_ids": [vectorstore_id]

            }
        }
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant_id)

    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

    message_content = messages[0].content[0].text
    message_value= message_content.value
    value = re.sub(r'【\d+:\d+†source】', '', message_value)
    return value
