# -*- coding: utf-8 -*-
from __future__ import (
    division,
    print_function,
)

import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import skimage
import selective_search

def main():

    # loading astronaut image
    #img = skimage.data.astronaut()
    img=skimage.io.imread('/data3/yczhang/dataset/DIOR-DCNet/JPEGImages-test/11726.jpg')
    print(type(img), img.size, img.shape)
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=800, sigma=0.8, min_size=5)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 100:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if h==0 or w==0 or w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    print(len(candidates))
    for x, y, w, h in candidates:
        #print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)


    method2=selective_search.selective_search(img, mode='fast', random_sort=True)
    refinedmethod2=selective_search.box_filter(method2, min_size=10, topN=2000)
    print(type(method2), len(method2), method2)
    print(type(refinedmethod2), len(refinedmethod2), refinedmethod2)
    for x1, y1, x2, y2 in refinedmethod2:
        bbox = mpatches.Rectangle(
            (x1, y1), (x2 - x1), (y2 - y1), fill=False, edgecolor='blue', linewidth=1)
        ax.add_patch(bbox)


    plt.show()

if __name__ == "__main__":
    main()


#regions[:10]
