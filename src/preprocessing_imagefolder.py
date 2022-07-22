import os
import cv2
import argparse


_RESIZE_MIN = 256


def resize_image(image):
    height, width, _ = image.shape
    new_height = height * _RESIZE_MIN // min(image.shape[:2])
    new_width = width * _RESIZE_MIN // min(image.shape[:2])
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def central_crop(image, crop_height, crop_width):
    height, width, _ = image.shape
    startx = width // 2 - (crop_width // 2)
    starty = height // 2 - (crop_height // 2)
    return image[starty:starty + crop_height, startx:startx + crop_width]


def preprocessing_images(img_path, output):
    fns = os.listdir(img_path)
    fns = [os.path.join(img_path, fn) for fn in fns if 'JPEG' in fn]
    for i in range(len(fns)):
        if i % 2000 == 0:
            print("%d/%d" % (i, len(fns)))
        # Load (as BGR)
        img = cv2.imread(fns[i])
        img = resize_image(img)
        img = central_crop(img, 224, 224)
        assert img.shape[0] == 224 and img.shape[1] == 224
        # Save (as RGB)
        # If it is in NP array, please revert last dim like [:,:,::-1]
        name, ext = os.path.splitext(os.path.basename(fns[i]))
        file_path = os.path.join(output, name + '.JPEG')
        cv2.imwrite(file_path, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, type=str, help="ImageNet image path")
    parser.add_argument('--output', '-o', type=str, default="", help="Output image path")
    args = parser.parse_args()
    if len(args.output) == 0:
        args.output = args.input
    print("Input dir:", args.input)
    print("Output dir:", args.output)
    import ipdb; ipdb.set_trace()
    preprocessing_images(args.input, args.output)