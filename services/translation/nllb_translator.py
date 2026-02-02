from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class NLLBTranslator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, text, src_lang="eng_Latn", tgt_lang="hin_Deva"):
        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(text, return_tensors="pt")
        generated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_lang]
        )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
