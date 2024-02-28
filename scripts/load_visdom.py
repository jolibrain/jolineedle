#!/usr/bin/python3

import sys
import os
import argparse
import logging

jn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
sys.path.append(jn_dir)
from src.visualizer import VisdomPlotter


def main():
    parser = argparse.ArgumentParser(
        description="Take visdom file and send it to the server"
    )
    parser.add_argument("file", help="Visdom file to reload")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Set logging level to INFO"
    )
    parser.add_argument("--env_name", type=str, help="Name of the environment")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if not args.env_name:
        args.env_name = os.path.basename(os.path.dirname(args.file))

    visdom = VisdomPlotter.load(args.file, args.env_name)
    visdom.update()


# ====

if __name__ == "__main__":
    main()
