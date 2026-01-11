import os
from typing import TextIO
from deep_translator import GoogleTranslator


def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}

    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(
            f"Expected one of {set(str2val.keys())}, got {string}")


def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(transcript: list, file: TextIO):
    for i, segment in enumerate(transcript, start=1):
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True)} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True)}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def translate_text(text: str, target_language: str) -> str:
    if not text or not text.strip():
        return text
    
    translator = GoogleTranslator(source='auto', target=target_language)
    return translator.translate(text)


def parse_timestamp(ts: str) -> float:
    hms, ms = ts.split(',')
    hours, minutes, seconds = hms.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms) / 1000.0


def translate_srt_file(src_path: str, dst_path: str, target_lang: str):
    segments = []
    with open(src_path, "r", encoding="utf-8") as f:
        block = []
        for line in f:
            line = line.rstrip('\n')
            if line.strip() == "":
                if block:
                    segments.append(block)
                    block = []
                continue
            block.append(line)
        if block:
            segments.append(block)

    parsed_segments = []
    for blk in segments:
        if len(blk) < 2:
            continue
        time_line = blk[1]
        if "-->" not in time_line:
            continue
        start_str, end_str = [t.strip() for t in time_line.split("-->")]
        text_lines = blk[2:] if len(blk) > 2 else [""]
        text = " ".join(text_lines).strip()
        parsed_segments.append({
            "start": parse_timestamp(start_str),
            "end": parse_timestamp(end_str),
            "text": translate_text(text, target_lang)
        })

    with open(dst_path, "w", encoding="utf-8") as srt:
        write_srt(parsed_segments, file=srt)

    print(f"Translated existing subtitles to {target_lang} at {dst_path}")
    return dst_path
