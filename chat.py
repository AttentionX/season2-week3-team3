import os
from typing import  List, Dict
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import openai
import logger
from vectorstore import query

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

MODEL="gpt-3.5-turbo"

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

def ChatMessage(role: Role, content: str) -> Dict:
    return {
        "role": role.value,
        "content": content,
    }
    

chat_history: List[ChatMessage] = [ChatMessage(
    role=Role.SYSTEM,
    content="You are a professor specialized in deep learning"
)]

RAG_PROMPT = """
\n######
Here are some relavant documents that may be helpful to answer the user.
The user doesn't know that these documents are provided to you.

{documents}
"""

DOC_TEMPLATE = """
Paper title: {title}
Page: {page}
Content: {content}
"""


def chat():
    logger.assistant("Hi! My name is professor. Ask me anything.")
    while True:
        user_input = input(">> ")
        # RAG
        temporary_chat_history = chat_history.copy()
        query_res = query(q=user_input)
        queried_docs = zip(query_res.documents, query_res.metadatas)
        documents_str = "\n".join([DOC_TEMPLATE.format(content=doc[0],title=doc[1]["title"], page=doc[1]["page"] ) for doc in queried_docs])
        temporary_chat_history.append(ChatMessage(
            role=Role.USER,
            content=user_input + RAG_PROMPT.format(documents=documents_str)
        ))
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages= temporary_chat_history,
        )["choices"][0]["message"]["content"]

        chat_history.append(ChatMessage(
            role=Role.USER,
            content=user_input
        )) # Append user input which is not modified with RAG to history
        chat_history.append(ChatMessage(
            role=Role.ASSISTANT,
            content=user_input
        ))
        logger.document(documents_str)
        logger.assistant(response)

chat()