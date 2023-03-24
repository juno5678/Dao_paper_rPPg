import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--my_bool', action='store_false', default=True)

args = parser.parse_args()

print(args.my_bool)