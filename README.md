# Paper to Speech (WIP)

## Setup
```sh
git clone git@github.com:chanwutk/paper-to-speech.git
cd paper-to-speech

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```sh
# see available options
python say.py --help

# output voices at <directory-containing-text-files-to-say>/*.wav
python say.py --dir <directory-containing-text-files-to-say>
```