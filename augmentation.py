import numpy as np
import random
import cv2
import glob
from PIL import Image
from matplotlib import cm


def generate(category_num, background, image_file_name, label_file_name):
    side = 512
    size = 512
    variation = 0.0003

    img = generate_origin_sticker(side)
    img = deformation_sticker(img, size, variation)

    # cv2.imshow('sticker', img)
    # cv2.waitKey(0)
    # cv2.destroyWindow('sticker')

    after_resize = img[0].size
    random_position_x = random.randrange(10, len(background) - after_resize - 10)
    random_position_y = random.randrange(10, len(background[0]) - after_resize - 10)

    background = paste_sticker(img, background, (random_position_x, random_position_y))

    # Change x y coordinate
    # 本是垂直為X，水平為Y，現在相反
    sticker_x, sticker_y, sticker_w, sticker_h = find_bounding_box(np.uint8(img), (random_position_x, random_position_y), 1)
    cv_x, cv_y, cv_w, cv_h = format_to_yolo_coordinate(background, sticker_x, sticker_y, sticker_w, sticker_h)

    save_label_file(label_file_name, category_num, cv_x, cv_y, cv_w, cv_h)
    cv2.imwrite(image_file_name, background)


def generate_origin_sticker(side):

    half_side = side // 2

    offset = 0 if side % 2 == 0 else 1

    img = np.zeros((side, side), np.uint8)
    x, y = np.meshgrid(np.arange(-half_side, half_side + offset), np.arange(-half_side, half_side + offset))
    img[x + y == int(0.5 * half_side)] = 1
    img[x - y == int(0.5 * half_side)] = 1
    img[x + y == int(-0.5 * half_side)] = 1
    img[x - y == int(-0.5 * half_side)] = 1

    img[x ** 2 + y ** 2 > int(0.5 * half_side) ** 2] = 0
    img[x ** 2 + y ** 2 < int(0.02 * half_side) ** 2] = 1

    kernel_size = int(0.01 * half_side)
    kernel = np.ones((kernel_size, kernel_size))

    img = cv2.morphologyEx(img * 255, cv2.MORPH_DILATE, kernel)

    return img


def deformation_sticker(img, size, variation):

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    box = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], np.float32)
    img = cv2.warpPerspective(
        np.float32(img),
        cv2.getPerspectiveTransform(
            box, np.float32(box * (1 + np.random.uniform(-variation, variation, size=box.shape)))
        ),
        (size, size)
    )

    center = (size // 2, size // 2)
    random_angle = random.randrange(0, 180)
    rotate_matrix = cv2.getRotationMatrix2D(center, random_angle, 1)
    img = cv2.warpAffine(img, rotate_matrix, (size, size))

    random_size = random.randrange(300, 800)
    random_thick = random.randrange(0, 2)
    img = cv2.resize(img, (random_size, random_size), interpolation=cv2.INTER_LINEAR)
    img = re_thick(img, random_thick)

    return img


def paste_sticker(sticker, background, position):

    for x in range(0, len(sticker)):
        for y in range(0, len(sticker[x])):
            if sticker[x][y] != 0:
                background[x + position[0]][y + position[1]] = [0, 0, 0]
    return background


def re_thick(image, iterations):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(image, kernel, iterations=iterations)
    return dilate


# find contour, then generate the bounding rectangle.
def find_bounding_box(uint8_img, position, padding=0):
    contours, _ = cv2.findContours(uint8_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = cv2.boundingRect(contours[0])

    x = x - padding + position[0]
    y = y - padding + position[1]
    w = w + padding
    h = h + padding

    return x, y, w, h


def format_to_yolo_coordinate(image, x, y, w, h):
    # swap
    temp = y
    y = x
    x = temp

    # yolo format
    image_width = len(image[0])
    image_height = len(image)
    cv_x = (x + w / 2) / image_width
    cv_y = (y + h / 2) / image_height
    cv_w = w / image_width
    cv_h = h / image_height

    return cv_x, cv_y, cv_w, cv_h


def save_label_file(file_name, category, x, y, w, h):
    label_file = open(file_name, 'a+')
    truncate = label_file.truncate()
    if truncate != 0:
        label_file.write('\n')
    label_file.write('%s %s %s %s %s' % (category, x, y, w, h))
    label_file.close()


def generate_file_names(product_path, file_name, count):
    new_file_name = file_name.split('/')
    new_file_name = new_file_name[len(new_file_name) - 1]
    new_file_name_list = []
    parts = new_file_name.split('.')

    for i in range(1, (count + 1)):
        suffix = "_%s" % i
        image_file_name = product_path + parts[0] + suffix + '.' + parts[1]
        label_file_name = image_file_name.split('.')[0] + '.txt'
        new_file_name_list.append([image_file_name, label_file_name])

    return new_file_name_list


if __name__ == '__main__':
    SOURCE_PATH = 'sources/*.JPG'
    PRODUCT_PATH = 'products/'
    PRODUCTION_MULTIPLE = 10
    CLASS_NUMBER = 0

    count = 1

    file_names = glob.glob(SOURCE_PATH)
    for file_name in file_names:
        file_name_list = generate_file_names(PRODUCT_PATH, file_name, PRODUCTION_MULTIPLE)
        for (image_file_name, label_file_name) in file_name_list:
            background = cv2.imread(file_name)
            generate(CLASS_NUMBER, background, image_file_name, label_file_name)
            percent = count / (len(file_names) * PRODUCTION_MULTIPLE) * 100
            percent = int(percent)
            print('[%s%%] Output Image: ' % percent + image_file_name)
            count += 1
