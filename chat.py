import os
from typing import  List, Dict
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import openai
import logger

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


def chat():
    logger.assistant("Hi! My name is professor. Ask me anything.")
    while True:
        user_input = input(">> ")
        chat_history.append(ChatMessage(
            role=Role.USER,
            content=user_input
        ))
        # TODO: RAG input
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages= chat_history,
        )["choices"][0]["message"]["content"]
        chat_history.append(ChatMessage(
            role=Role.ASSISTANT,
            content=user_input
        ))
        # TODO: Show retrieved document
        document = ""
        logger.document(document)
        logger.assistant(response)

chat()