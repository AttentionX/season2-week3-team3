from typing import Any, List
from dataclasses import dataclass
from enum import Enum
import openai

MODEL="gpt-3.5-turbo"

class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class ChatMessage:
    role: Role
    content: str

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return {
            "role": self.role,
            "content": self.content,
        }
    

chat_history: List[ChatMessage] = [ChatMessage(
    
)]



def chat():
    openai.ChatCompletions.create(model=MODEL)

