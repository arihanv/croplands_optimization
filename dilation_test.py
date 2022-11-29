# An animation to test out different dilations on the ground truth image
import numpy as np
import matplotlib.pyplot as plt
import cv2

ground_truth = plt.imread('./best_map.png').mean(axis=-1)

fig,ax = plt.subplots(1,2)
ax[0].imshow(ground_truth,'Greys_r')
for i in range(1, 25):
    kernel = np.ones((i, i), np.uint8)
    img_dilation = cv2.dilate(ground_truth, kernel, iterations=1)
    ax[1].cla()
    print(i)
    plt.title(f"Dilation with kernel of {i} * {i}")
    ax[1].imshow(img_dilation,'Greys_r')
    plt.show(block=False)
    plt.pause(0.25)

