import os
import argparse
from typing import TYPE_CHECKING


import numpy as np

if TYPE_CHECKING:
    from torch.multiprocessing import Queue


def process_text(text: str):
    return "\n\n".join(p.replace("\n", " ") for p in text.split("\n\n"))


def chunks(lst, n):
    n = len(lst) // n
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def split_text(text: str, chunk_count: int) -> list[str]:
    from TTS.utils.synthesizer import Synthesizer

    segmenter = Synthesizer._get_segmenter('en')
    sens = segmenter.segment(text)
    assert isinstance(sens, list)
    return [
        "\n".join(chunk)
        for chunk
        in chunks(sens, chunk_count)
    ]


def tts_chunk(
    rank: int,
    texts: list[str],
    queues: "list[Queue]",
    emotion: str,
    speaker: str | None = None,
    speaker_wav: str | None = None
):
    import torch

    torch.cuda.set_device(rank)
    queue = queues[rank]
    text = texts[rank]

    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer

    # print('Initializing Model')
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

    # print('Generating Voice')
    if speaker_wav is not None:
        queue.put(tts.tts(
            text=text,
            language="en",
            emotion=emotion,
            speaker_wav=speaker_wav,
        ))
    else:
        queue.put(tts.tts(
            text=text,
            language="en",
            emotion=emotion,
            speaker=speaker or "Tanja Adelina",
        ))
    synthesizer = tts.synthesizer
    assert isinstance(synthesizer, Synthesizer)
    queue.put(synthesizer.output_sample_rate)
    return


def tts_to_file(text: str, output_path: str, emotion: str, speaker: str | None = None, speaker_wav: str | None = None):
    text = process_text(text)

    import torch
    import torch.multiprocessing as mp
    import torch.multiprocessing.spawn
    from TTS.utils.audio.numpy_transforms import save_wav

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
    else:
        device_count = 1
    torch.cuda.empty_cache()

    qs = [mp.Queue(2) for _ in range(device_count)]
    texts = split_text(text, device_count)
    context = torch.multiprocessing.spawn(
        tts_chunk,
        args=(texts, qs, emotion, speaker, speaker_wav),
        nprocs=device_count,
        join=False
    )

    wav: list[int] = []
    sample_rate = None
    for q in qs:
        w, _sample_rate = q.get(), q.get()
        wav += list(w) + [0] * 10000

        if sample_rate is None:
            sample_rate = _sample_rate
        else:
            assert sample_rate == _sample_rate, (sample_rate, _sample_rate)
    
    assert context is not None
    context.join(1)

    assert isinstance(sample_rate, int)
    save_wav(wav=np.array(wav), path=output_path, sample_rate=sample_rate)


def main():
    parser = argparse.ArgumentParser(description='Say')
    
    parser.add_argument('-s', '--speaker', type=str, default=None, required=False,
                        help='Speaker to say. \nSee options and examples in ./voices. Replace the name "-" with " " (space) and without ".wav"')
    parser.add_argument('-e', '--emotion', type=str, default=None, required=False,
                        help='Emotion of the speaker')
    parser.add_argument('-t', '--text', type=str, default=None, required=False,
                        help='Text to say')
    parser.add_argument('-d', '--dir', type=str, default=None, required=False,
                        help='Directory of texts to say')
    parser.add_argument('-o', '--output', type=str, default='output.wav', required=False,
                        help='Output voice file (will be ignored if dir specified)')
    
    args = parser.parse_args()

    if args.text is not None and args.dir is not None:
        raise Exception('Please only select either text or dir.')

    if args.text is None:
        args.text = "Spatialyze: A Geospatial Video Analytics System with Spatial-Aware Optimizations. " \
            "Videos that are shot using commodity hardware such as phones and surveillance cameras record various metadata such as time and location. " \
            "We encounter such geospatial videos on a daily basis and such videos have been growing in volume significantly. " \
            "Yet, we do not have data management systems that allow users to interact with such data effectively. " \
            "In this paper, we describe Spatialyze, a new framework for end to end querying of geospatial videos. " \
            "Spatialyze comes with a domain specific language where users can construct geospatial video analytic workflows using a 3 step, declarative, build-filter-observe paradigm. " \
            "Internally, Spatialyze leverages the declarative nature of such workflows, the temporal-spatial metadata stored with videos, and physical behavior of real-world objects to optimize the execution of workflows. " \
            "Our results using real world videos and workflows show that Spatialyze can reduce execution time by up to 5.3 time, while maintaining up to 97.1% accuracy compared to unoptimized execution."

    import torch


    torch.multiprocessing.set_start_method('spawn')
    if args.dir is None:
        tts_to_file(args.text, args.output, args.emotion, args.speaker)
    else:
        for filename in sorted(os.listdir(args.dir)):
            path = os.path.join(args.dir, filename)
            with open(path, 'r') as f:
                tts_to_file(f.read(), path + '.wav', args.emotion, args.speaker)


if __name__ == '__main__':
    main()