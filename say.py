import argparse
import os
import random
from multiprocessing import Process
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
    input_queues: "list[Queue]",
    output_queues: "list[Queue]",
    emotion: str,
    speaker: str | None = None,
    speaker_wav: str | None = None
):
    import torch

    input_queue = input_queues[rank]
    output_queue = output_queues[rank]

    from TTS.api import TTS
    from TTS.utils.synthesizer import Synthesizer

    # print('Initializing Model')
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    synthesizer = tts.synthesizer
    assert isinstance(synthesizer, Synthesizer)

    while True:
        input_values = input_queue.get()
        if input_values is None:
            break

        text, output_path, idx = input_values

        # print('Generating Voice')
        if speaker_wav is not None:
            wav = tts.tts(
                text=text,
                language="en",
                emotion=emotion,
                speaker_wav=speaker_wav,
            )
        else:
            wav = tts.tts(
                text=text,
                language="en",
                emotion=emotion,
                speaker=speaker or "Tanja Adelina",
            )
        output_queue.put((wav, synthesizer.output_sample_rate, output_path, idx))
    
    output_queue.put(None)


def waves_to_file(output_queues: "list[Queue]"):
    while True:
        waves: list[tuple[int, list[int]]] = []
        sample_rate = None
        output_path = None
        output_values = [q.get() for q in output_queues]

        if all(v is None for v in output_values):
            return

        assert all(v is not None for v in output_values)
        for w, _sample_rate, _output_path, _idx in output_values:
            if output_path is None:
                output_path = _output_path
            assert _output_path == output_path, (output_path, _output_path)

            waves.append((_idx, w))

            if sample_rate is None:
                sample_rate = _sample_rate
            assert sample_rate == _sample_rate, (sample_rate, _sample_rate)
        
        wav: list[int] = []
        for _, w in sorted(waves):
            wav += list(w) + [0] * 10000
        
        from TTS.utils.audio.numpy_transforms import save_wav
        assert isinstance(sample_rate, int)
        assert isinstance(output_path, str)
        save_wav(wav=np.array(wav), path=output_path, sample_rate=sample_rate)


class TTSToFile:
    def __init__(
        self,
        emotion: str | None = None,
        speaker: str | None = None,
        speaker_wav: str | None = None
    ):
        import torch
        import torch.multiprocessing as mp

        if torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()
        else:
            self.device_count = 1
        torch.cuda.empty_cache()

        self.output_queues = [mp.Queue() for _ in range(self.device_count)]
        self.input_queues = [mp.Queue() for _ in range(self.device_count)]
        context = mp.spawn(
            tts_chunk,
            args=(self.input_queues, self.output_queues, emotion, speaker, speaker_wav),
            nprocs=self.device_count,
            join=False
        )
        assert context is not None
        self.context = context
        self.writer = Process(target=waves_to_file, args=(self.output_queues,))
        self.writer.start()
    
    def tts_to_file(self, text: str, output_path: str):
        texts = split_text(text, self.device_count)
        for idx, (text, queue) in enumerate(zip(
            texts,
            random.sample(self.input_queues, len(self.input_queues))
        )):
            queue.put((text, output_path, idx))
    
    def close(self):
        for q in self.input_queues:
            q.put(None)
        self.context.join(1)

        for q in self.input_queues:
            q.close()

        self.writer.join()
        self.writer.kill()
        self.writer.terminate()
        
        for q in self.output_queues:
            q.close()


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
    tts = TTSToFile(args.emotion, args.speaker)
    if args.dir is None:
        tts.tts_to_file(args.text, args.output)
    else:
        for filename in sorted(os.listdir(args.dir)):
            if not filename.endswith('.wav'):
                path = os.path.join(args.dir, filename)
                with open(path, 'r') as f:
                    tts.tts_to_file(f.read(), path + '.wav')
    
    tts.close()


if __name__ == '__main__':
    main()