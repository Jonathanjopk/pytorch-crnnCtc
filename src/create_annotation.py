import argparse
import glob
import os

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-image_path', type=str, default=None, help='image input path')
parser.add_argument('-anno_path', type=str, default=None, help='annotation output path')
parser.add_argument('-valid_ratio', type=float, default=None, help='ratio for validation dataset. if not given, then it will be testing mode')
args = parser.parse_args()


def main():
    image_list = glob.glob(os.path.join(args.image_path, "*"))

    output = []
    for path in image_list:
        output.append((path, path.split('#')[-1].split('.')[0])) # get lebel from filename pattern

    if args.valid_ratio:
        anno_train, anno_valid = train_test_split(output, test_size=args.valid_ratio)
        with open(os.path.join(args.anno_path, 'annotation_train.txt'), 'w') as f:
            for ele in anno_train:
                f.write(ele[0] + ' ' + ele[1] + '\n')

        with open(os.path.join(args.anno_path, 'annotation_val.txt'), 'w') as f:
            for ele in anno_valid:
                f.write(ele[0] + ' ' + ele[1] + '\n')

    else:
        with open(os.path.join(args.anno_path, 'annotation_test.txt'), 'w') as f:
            for ele in output:
                f.write(ele[0] + ' ' + ele[1] + '\n')


if __name__ == "__main__":
    main()
