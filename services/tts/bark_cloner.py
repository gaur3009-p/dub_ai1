import uuid
import soundfile as sf
from bark import generate_audio, preload_models

# Load Bark models once
preload_models()


class BarkVoiceCloner:
    def synthesize(
        self,
        text: str,
        history_prompt: str,
        language: str
    ) -> str:
        """
        history_prompt: path to enrolled speaker wav
        language: en / hi / fr / es
        """

        # Bark language conditioning
        lang_token = {
            "en": "[en]",
            "hi": "[hi]",
            "fr": "[fr]",
            "es": "[es]",
        }.get(language, "[en]")

        text = f"{lang_token} {text}"

        audio = generate_audio(
            text,
            history_prompt=history_prompt
        )

        output_path = f"/tmp/{uuid.uuid4()}.wav"
        sf.write(output_path, audio, 24000)
        return output_path
