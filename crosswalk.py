import cv2
import numpy as np

global brightness, contrast, block_size, constant, image_original, image_segmented, filtered_contours
SEGMENTATION_COLOR = (0, 0, 255)


def log():
    global brightness, contrast, block_size, constant
    print("Brightness:\t\t" + str(brightness))
    print("Contrast:\t\t" + str(contrast))
    print("Block size:\t\t" + str(block_size))
    print("Constant:\t\t" + str(constant))
    print("--------------------")


def on_change_brightness(val):
    global brightness

    brightness = val - 255
    show_image_window()
    log()


def on_change_contrast(value):
    global contrast

    contrast = value - 255
    show_image_window()
    log()


def on_change_constant(value):
    global constant, block_size
    constant = value - 50
    make_segmentation()
    log()


def on_change_block_size(value):
    global block_size

    block_size = value

    if value < 3:
        block_size = 3
    elif value % 2 == 0:
        block_size += 1

    make_segmentation()
    log()


def show_image_window():
    global image_prepared, image_segmented, brightness, contrast

    factor = (259 * (contrast + 255)) / (255 * (259 - contrast))

    x = np.arange(0, 256, 1)
    lut = np.uint8(np.clip(brightness + factor * (np.float32(x) - 128.0) + 128, 0, 255))
    image_segmented = cv2.LUT(image_prepared, lut)
    cv2.imshow('Image', image_segmented)


def make_segmentation():
    global image_segmented, block_size, constant, filtered_contours

    image_grayscale = cv2.cvtColor(image_segmented, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(image_grayscale, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block_size, constant)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_segmented_display = image_segmented.copy()

    filtered_contours = []
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])

        if area > 1000:
            filtered_contours.append(contours[i])

            cv2.drawContours(image_segmented_display, contours, i, SEGMENTATION_COLOR, -1, cv2.LINE_4)
            cv2.imshow('Prepared image', image_segmented_display)


def show_segmentation_window():
    global image_segmented, image_original

    cv2.namedWindow('Prepared image')

    cv2.createTrackbar('Block size', 'Prepared image', 0, 100, on_change_block_size)
    cv2.createTrackbar('Constant', 'Prepared image', 0, 100, on_change_constant)

    cv2.imshow('Prepared image', image_segmented)


def show_segmented_image():
    global image_segmented, image_original, filtered_contours

    cv2.namedWindow('Prepared image')

    cv2.createTrackbar('Block size', 'Prepared image', 0, 100, on_change_block_size)
    cv2.createTrackbar('Constant', 'Prepared image', 0, 100, on_change_constant)

    image_original_display = image_original.copy()
    for i in range(0, len(filtered_contours)):
        cv2.drawContours(image_original_display, filtered_contours, i, SEGMENTATION_COLOR, -1, cv2.LINE_AA)

    cv2.imshow('Segmented image', image_original_display)


def init():
    global brightness, contrast, block_size, constant, filtered_contours

    filtered_contours = []
    brightness = contrast = 0
    block_size = 3
    constant = 1


def on_change_image(value):
    global image_original, image_prepared

    value += 1

    image_original = cv2.imread("images/crosswalk" + str(value) + ".jpg", cv2.IMREAD_COLOR)
    image_prepared = image_original.copy()
    show_image_window()


if __name__ == '__main__':
    init()

    cv2.namedWindow('Image')
    image_original = cv2.imread("images/crosswalk1.jpg", cv2.IMREAD_COLOR)
    image_prepared = image_original.copy()

    cv2.createTrackbar('Image', 'Image', 0, 6, on_change_image)
    cv2.createTrackbar('Contrast', 'Image', 255, 511, on_change_contrast)
    cv2.createTrackbar('Brightness', 'Image', 255, 511, on_change_brightness)

    cv2.imshow('Image', image_prepared)
    show_image_window()

    while True:
        ch = cv2.waitKey()
        if ch == 27 or ch == ord('q'):
            break
        if ch == ord('p'):
            show_segmentation_window()
        if ch == ord('s'):
            show_segmented_image()

    cv2.destroyAllWindows()
