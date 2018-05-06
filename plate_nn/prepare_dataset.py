import argparse
import os
import random

parser = argparse.ArgumentParser(description='Split dataset into training and validation')
parser.add_argument('data_dir', type=str, help='Dataset directory')
parser.add_argument('percentage', default=0.1, help='Validation percentage', type=float)

args = parser.parse_args()

dataset = [
    os.path.join(root, f) + '\n'
    for root, dirs, files in os.walk(args.data_dir)
    for f in files
    if f.lower().endswith('.png') or f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]

random.shuffle(dataset)

val_count = int(len(dataset) * float(args.percentage))

with open(args.data_dir + '/val.txt', 'w') as f:
    f.writelines(dataset[:val_count])

with open(args.data_dir + '/train.txt', 'w') as f:
    f.writelines(dataset[val_count:])

print('Train examples %s, validation examples %s' % (len(dataset) - val_count, val_count))
