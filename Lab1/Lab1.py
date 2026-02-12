import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

#Task 2a

img = cv2.imread('Otter.png')

if img is None:
    print("Error: Could not load Otter.png")
else:
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(img_rgb)
    plt.title("OpenCV Image")
    plt.axis("off")
    plt.show()

#Task 2b

img2 = Image.open('Otter.png')

plt.figure()
plt.imshow(img2, cmap=cm.Greys_r)
plt.title("Image")
plt.axis("off")
plt.show()

#Task 3a

cv2.imwrite('Monkey.png', img)

#Task 3b

img2.save('Monkey.png')

#Task 3c

print("OpenCV Image Shape:")
print(img.shape)

print("OpenCV Image Array:")
print(img)


img_array = np.array(img2)

print("PIL Image Shape:")
print(img_array.shape)

print("PIL Image Array:")
print(img_array)
