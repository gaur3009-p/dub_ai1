from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class NLLBTranslator:
    def __init__(self, model_name: str):
        # IMPORTANT: use_fast=False for NLLB
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text or not text.strip():
            raise ValueError("Empty text received for translation")

        # Prefix source language (MANDATORY for NLLB)
        text = f"<{src_lang}> {text}"

        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                    f"<{tgt_lang}>"
                ),
                max_length=256,
            )

        translated = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]

        return translated.strip()
