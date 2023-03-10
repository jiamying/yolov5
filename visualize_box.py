import argparse
import os

import cv2

from utils.plots import Annotator, colors


names = ['text', 'image']


def normed2xyxy(normed, w, h):
    return [(normed[0] - normed[2] / 2) * w,
            (normed[1] - normed[3] / 2) * h,
            (normed[0] + normed[2] / 2) * w,
            (normed[1] + normed[3] / 2) * h,]

def visualize(image_path, label_path, save_path):
    img = cv2.imread(image_path)
    annotator = Annotator(img)
    with open(label_path, 'r') as fin:
        for line in fin.readlines():
            line = line.split()
            cls = line[0]
            xyxy = normed2xyxy(list(map(float, line[1:])), img.shape[1], img.shape[0])
            c = int(cls)  # integer class
            label = names[c]
            annotator.box_label(xyxy, label, color=colors(c, True))

    img_out = annotator.result()
    cv2.imwrite(save_path, img_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='path to an image or a directory of images')
    parser.add_argument('--label', type=str, help='path to a label file or a directory of labels')
    parser.add_argument('--output', type=str, help='directory to save the annotated images')
    args = parser.parse_args()

    assert os.path.exists(args.image) and os.path.exists(args.label)
    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.image):
        image_path = args.image
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        if os.path.isfile(args.label):
            label_path = args.label
        else:
            label_path = os.path.join(args.label, image_name + '.txt')
        if not os.path.exists(label_path):
            print(f"Label can't be found in {label_path}. Skipped.")
            return
        save_path = os.path.join(args.output, image_name + '.jpg')

        visualize(image_path, label_path, save_path)

    else:
        assert os.path.isdir(args.label)
        image_list = os.listdir(args.image)
        for image in image_list:
            image_path = os.path.join(args.image, image)
            image_name = os.path.splitext(image)[0]
            label_path = os.path.join(args.label, image_name + '.txt')
            if not os.path.exists(label_path):
                print(f"Label can't be found in {label_path}. Skipped.")
                continue
            save_path = os.path.join(args.output, image_name + '.jpg')
            visualize(image_path, label_path, save_path)


if __name__ == '__main__':
    main()
