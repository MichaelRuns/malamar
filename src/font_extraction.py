import os
from PIL import Image
import numpy as np
import cv2

def get_upper_case_map():
    return [
    "ABCDEFGHI",
    "JKLMNOPQR",
    "STUVWXYZ",
    "x():;[]<>",
    "-?!%^\*,&"
    ]

def get_lower_case_map():
    return [
        "abcdefghi",
        "jklmnopqr",
        "stuvwxyz",
        "x():;[]<>",
        "-?!%^\*,&"
    ]


def extract_font_templates(source_image_path, output_dir="assets/font/templates", upper=False):
    char_map = get_lower_case_map()
    if upper:
        char_map = get_upper_case_map()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    img = Image.open(source_image_path).convert('L')
    grid = np.array(img)
    start_y, start_x = 0, 0
    row_height, col_width = 62, 62
    for row, row_str in enumerate(char_map):
        slice_y = start_y + (row*row_height)
        for col, char in enumerate(row_str):
            slice_x = start_x + (col*col_width)
            slice = grid[slice_y:slice_y+row_height, slice_x:slice_x+col_width]
            cv2.imshow(char, slice)
            cv2.waitKey(100)
            image = Image.fromarray(slice)
            print(char)
            image.save(output_dir+"/"+char+".png")

def main():
    extract_font_templates("assets/font/font_uppercase.png", "assets/font/templates", True)
    extract_font_templates("assets/font/font_lowercase.png", "assets/font/templates", False)
    
if __name__ == '__main__':
    main()

    