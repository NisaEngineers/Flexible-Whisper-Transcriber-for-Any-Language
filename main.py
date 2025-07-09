# main.py
import os
import argparse
import unicodedata
import re
from num2words import num2words

import whisper
import pandas as pd
from datasets import Dataset, Audio

def normalize_text(text: str, lang: str = None) -> str:
    """
    Generic text normalizer.
    - Unicode NFC
    - Collapse whitespace
    - Optionally expand numbers for supported languages
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Example: expand integers for Italian and English
    if lang in ("it", "en"):
        def repl(m):
            s = m.group()
            return num2words(int(s), lang=lang) if s.isdigit() else s
        text = re.sub(r'\b\d+\b', repl, text)

    return text

def transcribe_files(
    audio_paths,
    model_name="base",
    language=None,
    output_csv="transcripts.csv",
    output_train_list="train_list.txt",
    normalize=True
):
    model = whisper.load_model(model_name)
    records = []
    train_lines = []

    for path in audio_paths:
        # load and get duration
        audio = whisper.load_audio(path)
        duration = audio.shape[-1] / whisper.audio.SAMPLE_RATE

        # transcribe (auto-detect if language is None)
        args = {"language": language} if language else {}
        result = model.transcribe(path, **args)
        txt = result["text"].strip()
        norm_txt = normalize_text(txt, language) if normalize else txt

        records.append({
            "audio": path,
            "duration_s": round(duration, 2),
            "raw_text": txt,
            "norm_text": norm_txt,
            "length": len(norm_txt)
        })
        # For TTS train list: filename|text|speaker_id
        fname = os.path.basename(path)
        train_lines.append(f"{fname}|{norm_txt}|0")

        print(f"{fname}: {len(norm_txt)} chars")

    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved transcripts to {output_csv}")

    # Save train_list.txt
    with open(output_train_list, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))
    print(f"Saved TTS list to {output_train_list}")

    # Optional: Hugging Face dataset
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("audio", Audio())
    ds.to_parquet(output_csv.replace(".csv", ".parquet"))
    print("Parquet dataset saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Whisper-based multi-language transcriber"
    )
    parser.add_argument(
        "--model", "-m", default="large", help="Whisper model size"
    )
    parser.add_argument(
        "--lang", "-l",
        help="Language code (e.g. it, en). Omit for auto-detect"
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="One or more audio file paths (wav, mp3, m4a, etc.)"
    )
    parser.add_argument(
        "--no-normalize", action="store_false",
        dest="normalize", help="Skip text normalization"
    )
    args = parser.parse_args()

    transcribe_files(
        audio_paths=args.inputs,
        model_name=args.model,
        language=args.lang,
        normalize=args.normalize
    )
