import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from typing import Callable
from .utils import filename, str2bool, write_srt, translate_text, translate_srt_file


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--model", default="small",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--output_srt", type=str2bool, default=False,
                        help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
    help="What is the origin language of the video? If unset, it is detected automatically.")
    parser.add_argument("--translate_to", type=str, default=None,
                        help="translate subtitles to target language (e.g., 'zh-CN', 'ja', 'ko').")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    translate_to: str = args.pop("translate_to")
    videos = args.pop("video")
    
    os.makedirs(output_dir, exist_ok=True)

    subtitles_original = {}
    subtitles_translated = {}
    embed_map = {}  # path -> srt path to embed
    videos_to_transcribe = []
    translate_existing = []  # (path, original_srt, translated_srt)
    srt_base_dir = output_dir if output_srt or srt_only else tempfile.gettempdir()

    for path in videos:
        srt_original = os.path.join(srt_base_dir, f"{filename(path)}.srt")
        original_valid = os.path.exists(srt_original) and os.path.getmtime(srt_original) >= os.path.getmtime(path)
        if original_valid:
            subtitles_original[path] = srt_original
            print(f"Using cached original subtitles for {filename(path)} from {srt_original}")

        translated_valid = False
        srt_translated = None
        if translate_to:
            srt_translated = os.path.join(srt_base_dir, f"{filename(path)}_{translate_to}.srt")
            translated_valid = os.path.exists(srt_translated) and os.path.getmtime(srt_translated) >= os.path.getmtime(path)
            if translated_valid:
                subtitles_translated[path] = srt_translated
                print(f"Using cached translated subtitles for {filename(path)} from {srt_translated}")

        if translate_to:
            if translated_valid:
                embed_map[path] = srt_translated
            elif original_valid:
                translate_existing.append((path, srt_original, srt_translated))
            else:
                videos_to_transcribe.append(path)
        else:
            if original_valid:
                embed_map[path] = srt_original
            else:
                videos_to_transcribe.append(path)

    if translate_existing:
        for path, src_srt, dst_srt in translate_existing:
            translated_path = translate_srt_file(src_srt, dst_srt, translate_to)
            if translated_path:
                subtitles_translated[path] = translated_path
                embed_map[path] = translated_path
            else:
                videos_to_transcribe.append(path)

    if videos_to_transcribe:
        if model_name.endswith(".en"):
            warnings.warn(
                f"{model_name} is an English-only model, forcing English detection.")
            args["language"] = "en"
        elif language != "auto":
            args["language"] = language
            
        model = whisper.load_model(model_name)
        audios = get_audio(videos_to_transcribe)
        new_original, new_translated = get_subtitles(
            audios, output_srt or srt_only, output_dir, lambda audio_path: model.transcribe(audio_path, **args),
            translate_to=translate_to
        )
        subtitles_original.update(new_original)
        subtitles_translated.update(new_translated)

        for path in videos_to_transcribe:
            if translate_to and path in new_translated:
                embed_map[path] = new_translated[path]
            else:
                embed_map[path] = new_original.get(path)

    if srt_only:
        return

    for path in videos:
        srt_to_embed = embed_map.get(path) or subtitles_translated.get(path) or subtitles_original.get(path)
        if not srt_to_embed:
            print(f"Error: No subtitle found for {filename(path)}, skipping...")
            continue

        out_path = os.path.join(output_dir, f"{filename(path)}.mp4")

        print(f"Adding subtitles to {filename(path)}...")

        video = ffmpeg.input(path)
        audio = video.audio

        ffmpeg.concat(
            video.filter('subtitles', filename=srt_to_embed, force_style="OutlineColour=&H40000000,BorderStyle=1,Outline=0.5"), audio, v=1, a=1
        ).output(out_path).run(quiet=True, overwrite_output=True)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")


def get_audio(paths):
    temp_dir = tempfile.gettempdir()

    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        ffmpeg.input(path).output(
            output_path,
            acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True, overwrite_output=True)

        audio_paths[path] = output_path

    return audio_paths


def get_subtitles(audio_paths: dict, output_srt: bool, output_dir: str, transcribe: Callable, translate_to: str | None = None):
    subtitles_original = {}
    subtitles_translated = {}

    for path, audio_path in audio_paths.items():
        srt_base_dir = output_dir if output_srt else tempfile.gettempdir()
        srt_original = os.path.join(srt_base_dir, f"{filename(path)}.srt")
        
        print(
            f"Generating subtitles for {filename(path)}... This might take a while."
        )

        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        with open(srt_original, "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
        subtitles_original[path] = srt_original
        print(f"Saved original subtitles to {srt_original}")

        if translate_to:
            import copy
            translated_segments = copy.deepcopy(result["segments"])
            print(f"Translating subtitles to {translate_to}...")
            for segment in translated_segments:
                segment["text"] = translate_text(segment["text"], translate_to)

            srt_translated = os.path.join(srt_base_dir, f"{filename(path)}_{translate_to}.srt")
            with open(srt_translated, "w", encoding="utf-8") as srt:
                write_srt(translated_segments, file=srt)
            subtitles_translated[path] = srt_translated
            print(f"Saved translated subtitles to {srt_translated}")

    return subtitles_original, subtitles_translated


if __name__ == '__main__':
    main()
