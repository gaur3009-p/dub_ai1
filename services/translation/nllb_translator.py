from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class NLLBTranslator:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        prefix = f"<{src_lang}>"
        inputs = self.tokenizer(prefix + text, return_tensors="pt")

        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(f"<{tgt_lang}>")
        )

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return decoded[0]
