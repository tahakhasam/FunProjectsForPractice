import os, sys
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
from itertools import product
from functools import reduce
import math
import numpy as np
import argparse

def parse_cli_args():
    parser = argparse.ArgumentParser(prog="mosaic image builder")
    parser.add_argument(
        "--image", 
        "-i", 
        dest="main_image", 
        type=str, 
        nargs=1, 
        metavar="MAIN IMAGE PATH", 
        help="Please enter the path of the image that is meant to be turned into a mosaic image."
    )
    parser.add_argument(
        "--source",
        "-s",
        dest="source_path",
        type=str,
        nargs=1,
        metavar="SOURCE IMAGE PATH",
        help="Please enter the path of the image directory which will be used to fill up the mosaic image."
    )
    parser.add_argument(
        "--methods",
        "-m",
        dest="methods",
        nargs="*",
        default="mean",
        metavar="METHOD",
        help="Please enter the method of distance formula to be used when building mosaic image [mean/median/harmonic_mean/geometric_mean]."
    )
    return parser.parse_args(args=None if sys.argv[1:] else ['--help'])


def resize_the_source_image(source_image):
    image = Image.open(source_image)
    max_size = (max(image.size) // 100) * 100
    new_shape = (max_size, max_size)
    resized_image = image.resize(new_shape)
    w, h = (50, 50)
    image_w, image_h = new_shape
    grid = create_grid(image_w, image_h, w, h)
    return resized_image, image_w, image_h, w, h, grid

def create_grid(image_w, image_h, box_w, box_h):
    grid = product(range(0, image_h-image_h%box_h, box_h), range(0, image_w-image_w%box_w, box_w))
    return grid

def find_distance(r1, g1, b1, r2, g2, b2):
    return math.sqrt(((r2 - r1) ** 2) + ((g2 - g1) ** 2) + ((b2 - b1) ** 2))

def find_average_colors(image):
    colors = image.getcolors(image.size[0] * image.size[1])
    average_colors = reduce(lambda x, y: ((x[0]+y[0])//2,((x[1][0]+y[1][0])//2,(x[1][1]+y[1][1])//2,(x[1][2]+y[1][2])//2)), colors)
    return average_colors[1]

def read_all_images_in_folder(images_path):
    print(f"Reading all images from {images_path}")
    image_entries = Path(images_path)
    return image_entries

def divide_into_4(image):
    w, h = image.size
    min_m = min([w // 2, h // 2])
    divide_grid = create_grid(w, h, min_m, min_m)
    measures = [] 
    for i, j in divide_grid:
        box = (j, i, j+min_m, i+min_m)
        section = image.crop(box)
        section_average = find_average_colors(section)
        measures.append(section_average)
    return measures

def fine_tune(main_average, input_average, method="mean"):
    distances = [find_distance(*ma, *ia) for ma, ia in zip(main_average, input_average)]
    if method == 'mean':
        return np.mean(distances)
    elif method == 'median':
        return np.median(distances)
    elif method == 'harmonic_mean':
        return len(distances) / np.sum(1 / np.array(distances))
    elif method == 'geometric_mean':
        return np.prod(distances) ** (1 / len(distances))

def create_mosaic_image(source_name, source_ext, images_path, source_image, method="mean"):
    all_images = [Image.open(imagepath) for imagepath in read_all_images_in_folder(images_path).iterdir()]
    main_image, image_w, image_h, w, h, grid = resize_the_source_image(source_image)
    all_images = [image.resize((w, h)) for image in all_images]
    all_average_colors = [divide_into_4(image) for image in all_images]
    print(f"Starting Mosaic image building of {source_name}{source_ext}")
    for i, j in grid:
        box = (j, i, j+w, i+h)
        section = main_image.crop(box)
        section_average = divide_into_4(section)
        min = None
        min_distance = None
        for index, average in enumerate(all_average_colors):
            distance = fine_tune(section_average, average, method)
            if min == None:
                min = index
                min_distance = distance
            elif distance < min_distance:
                min = index
                min_distance = distance
        main_image.paste(all_images[min], (j, i))
    print(f"Saving image {source_name}_mosaic_{method}{source_ext}")
    main_image.save(f"{source_name}_mosaic_{method}{source_ext}")

if __name__ == '__main__':
    args=parse_cli_args()
    source_image = args.main_image[0]
    images_path = args.source_path[0]
    source_path, source_ext = os.path.splitext(source_image)
    source_name = source_path.split("\\")[-1]
    for method in args.methods:
        create_mosaic_image(source_name, source_ext, images_path, source_image, method=method)
    
        

        



    