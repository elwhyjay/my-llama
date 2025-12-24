import os
from logging import getLogger
from typing import List

from sentencepiece import SentencePieceProcessor

logger = getLogger()

class Tokenizer:
    def __init__ (self,model_path:str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.sp = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Loaded SentencePiece model from {model_path}")

        self.number_words = self.sp.vocab_size()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        logger.info(
            f"Tokenizer initialized with vocab size: {self.number_words}, "
            f"BOS ID: {self.bos_id}, EOS ID: {self.eos_id}, PAD ID: {self.pad_id}"
        )
        assert self.sp.vocab_size == self.sp.get_piece_size(), "Vocabulary size mismatch"

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> List[int]:
        
        assert type(text) == str, "Input text must be a string"
        encoding_text = self.sp.model.encode(text)
        if bos:
            encoding_text = [self.bos_id] + encoding_text
        if eos:
            encoding_text =  [self.eos_id] + encoding_text
        return encoding_text
    

    def decode(self, token_ids: List[int]) -> str:

        assert type(token_ids) == list, "Input token_ids must be a list of integers"
        return self.sp.model.decode(token_ids)
