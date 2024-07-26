import multiprocessing
import os
import argparse


def process_text(text: str):
    return "\n\n".join(p.replace("\n", " ") for p in text.split("\n\n"))


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
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    print('Initializing Model')
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    def tts_to_file(text, output_path):
        tts.tts_to_file(
            text=process_text(text),
            language="en",
            file_path=output_path,
            emotion=args.emotion,
            speaker=args.speaker or "Tanja Adelina",
        )

    print('Generating Voice')
    if args.dir is None:
        tts_to_file(args.text, args.output)
    else:
        for filename in sorted(os.listdir(args.dir)):
            path = os.path.join(args.dir, filename)
            with open(path, 'r') as f:
                tts_to_file(f.read(), path + '.wav')


if __name__ == '__main__':
    main()