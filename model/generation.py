import json
import os
import sys
from pathlib import Path
from typing import List,Literal,Optional, Tuple, Union,TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import LLamaTransformer
from model.tokenizer import LLamaTokenizer
from model.attention import AttentionConfig

Role = Literal["system", "user", "assistant"]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

class Message(TypedDict):
    role: Role
    content: str

class CompletionPredict(TypedDict, total=False):
    generation: str
    tokens: List[str]
    logprobs: List[float]

class ChatPredict(TypedDict, total=False):
    generation: Message
    tokens: List[str]
    logprobs: List[float]

Dialog = List[Message]

class Llama:
    @staticmethod
    def build(

    ) -> "Llama":
        """
        build의 Docstring
        
        :return: 설명
        :rtype: Llama
        """
        