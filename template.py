import os
from itertools import product

import cv2
from matplotlib import pyplot as plt

photos_dir = 'photos1'
templates_dir = 'templates'

counter = 0


def show_fig(template, img, method):
    plt.subplot(121)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    plt.title('Detected')
    plt.xticks([])
    plt.yticks([])
    plt.suptitle(method)

    plt.show()


def save_fig(template, img, method):
    global counter
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(template, cmap="gray")
    ax1.set_title('Template')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img, cmap="gray")
    ax2.set_title('Detected')
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.suptitle(method)
    #fig.show()
    print(f'res/{counter}.jpg')
    fig.savefig(f'res/{counter}.jpg')
    counter += 1


def match(template_path, photo_path):
    methods = [
        'cv2.TM_CCOEFF',
        'cv2.TM_CCOEFF_NORMED',
        'cv2.TM_CCORR',
        'cv2.TM_CCORR_NORMED',
        'cv2.TM_SQDIFF',
        'cv2.TM_SQDIFF_NORMED',
    ]
    image = cv2.imread(photo_path, 0)
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]

    for method_name in methods:
        img = image.copy()
        method = eval(method_name)

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img, top_left, bottom_right, 0, 5)
        #show_fig(template, img, method)
        save_fig(template, img, method_name)


photos = map(lambda file_name: os.path.join(photos_dir, file_name), os.listdir(photos_dir))
templates = map(lambda file_name: os.path.join(templates_dir, file_name), os.listdir(templates_dir))

for (photo, template) in product(photos, templates):
    print(f'processing {template} and {photo}')
    match(template, photo)
