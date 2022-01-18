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
from tqdm.auto import tqdm
import argtyped


class Arguments(argtyped.Arguments):
    # input file such as captions/cap.X.X.json
    caption_file: Path
    # output captions
    output: Path
    # root image dir
    root: Path


def image2path(image_name: str, root_dir: Path):
    return root_dir / f"{image_name}.jpg"


if __name__ == "__main__":
    args = Arguments()
    print(args)

    with open(args.caption_file) as fid:
        data = json.load(fid)

    transformed = []
    err_counter = 0

    for item in tqdm(data):
        image1 = image2path(item["target"], args.root)
        image0 = image2path(item["candidate"], args.root)

        if not image1.is_file():
            # print(image1, "does not exist")
            err_counter += 1
            continue

        if not image0.is_file():
            # print(image0, "does not exist")
            err_counter += 1
            continue

        transformed.append(
            {
                "image1": str(image1),
                "image0": str(image0),
                "captions": item["captions"][0],
            }
        )

    print("# errors:", err_counter)
    print("# success:", len(transformed))

    with open(args.output, "w") as fid:
        json.dump(transformed, fid, indent=2)
