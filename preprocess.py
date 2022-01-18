"""
Preprocess captions/cap.X.X.json
each element is transformed from 
    target
    candidate
    captions
into:
    image1
    image0
    captions
"""
import json
from pathlib import Path
import argtyped


class Arguments(argtyped.Arguments):
    # input file such as captions/cap.X.X.json
    caption_file: Path
    # output file
    output: Path



if __name__ == "__main__":
    args = Arguments()

    with open(args.caption_file) as fid:
        data = json.load(fid)

    transformed = []
    for item in data:
        transformed.append({
            "image1": item['target'],
            "image0": item['candidate'],
            "captions": item['captions'][0],
        })

    with open(args.output, 'w') as fid:
        json.dump(transformed, fid, indent=2)
