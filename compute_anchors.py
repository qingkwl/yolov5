import argparse
import ast

import sklearn  # For avoid load error of some libraries

from src.autoanchor import kmean_anchors

def compute_anchors(args):
    anchors = kmean_anchors(args.dataset, args.num_anchors, args.img_size, args.threshold,
                            args.gen, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute anchors")
    parser.add_argument("--dataset", type=str, default="./config/data/coco.yaml", help="Path to dataset config file.")
    parser.add_argument("--num_anchors", type=int, default=9, help="The number of anchors.")
    parser.add_argument("--img_size", type=int, default=640, help="Image size.")
    parser.add_argument("--threshold", type=float, default=4.0,
                        help="Anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0")
    parser.add_argument("--gen", type=int, default=1000,
                        help="The number of generations to evolve anchors using genetic algorithm")
    parser.add_argument("--verbose", type=ast.literal_eval, default=True, help="Whether print verbose information.")
    args = parser.parse_args()
    compute_anchors(args)
