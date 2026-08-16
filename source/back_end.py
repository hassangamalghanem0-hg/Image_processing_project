import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# READ IMAGE

im_cv = cv2.imread("image.jpg")

if im_cv is None:
    print("Image not found!")
    exit()

cv2.imshow("Original Image", im_cv)

height, width, channels = im_cv.shape

print("Height  :", height)
print("Width   :", width)
print("Channels:", channels)

# COMPLEMENT

complement = 255 - im_cv
cv2.imshow("Complement Image", complement)

# RGB CHANNELS

blueImage = im_cv[:, :, 0]
greenImage = im_cv[:, :, 1]
redImage = im_cv[:, :, 2]

cv2.imshow("Blue Image", blueImage)
cv2.imshow("Green Image", greenImage)
cv2.imshow("Red Image", redImage)

# COLOR OPERATIONS

red_boost = im_cv.copy()
red_boost[:,:,2] = cv2.add(red_boost[:,:,2], 50)
cv2.imshow("Increase Red", red_boost)

swap_rg = im_cv.copy()
swap_rg[:,:,1], swap_rg[:,:,2] = im_cv[:,:,2], im_cv[:,:,1]
cv2.imshow("Swap R and G", swap_rg)

no_red = im_cv.copy()
no_red[:,:,2] = 0
cv2.imshow("No Red Channel", no_red)

# ROI

x, y, w, h = 100, 100, 200, 200
roi = im_cv[y:y+h, x:x+w]
cv2.imshow("ROI", roi)

# ARITHMETIC + LOGICAL

img2 = cv2.imread("image2.jpg")
img2 = cv2.resize(img2, (width, height))

cv2.imshow("Addition", cv2.add(im_cv, img2))
cv2.imshow("Subtraction", cv2.subtract(im_cv, img2))
cv2.imshow("Multiplication", cv2.multiply(im_cv, img2))
cv2.imshow("Division", cv2.divide(im_cv, img2))

cv2.imshow("AND", cv2.bitwise_and(im_cv, img2))
cv2.imshow("OR", cv2.bitwise_or(im_cv, img2))
cv2.imshow("XOR", cv2.bitwise_xor(im_cv, img2))

# GRAYSCALE + HISTOGRAM

gray = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray], [0], None, [256], [0,256])

plt.figure("Histogram")
plt.plot(hist)
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

# GAMMA CORRECTION

im = gray / 255.0

gamma = 2

correct = im ** gamma

correct = np.uint8(correct * 255)

cv2.imshow("Gamma Corrected Image", correct)

# HISTOGRAM STRETCHING + EQUALIZATION

min_val = np.min(gray)
max_val = np.max(gray)

stretch = ((gray - min_val) / (max_val - min_val)) * 255

stretch = np.uint8(stretch)

cv2.imshow("Histogram Stretching", stretch)

equalized = cv2.equalizeHist(gray)
cv2.imshow("Histogram Equalization", equalized)

# FILTERS

cv2.imshow("Average Filter", cv2.blur(im_cv, (5,5)))
cv2.imshow("Median Filter", cv2.medianBlur(im_cv, 5))

rank_kernel = np.ones((3,3), np.uint8)
rank_order = cv2.morphologyEx(gray, cv2.MORPH_OPEN, rank_kernel)
cv2.imshow("Rank Order Filter", rank_order)

# Non-linear
cv2.imshow("Max Filter", cv2.dilate(gray, np.ones((3,3), np.uint8)))
cv2.imshow("Min Filter", cv2.erode(gray, np.ones((3,3), np.uint8)))

def mode_filter(img):
    h, w = img.shape
    output = np.zeros_like(img)
    for i in range(1, h-1):
        for j in range(1, w-1):
            window = img[i-1:i+2, j-1:j+2].flatten()
            output[i,j] = stats.mode(window, keepdims=True)[0]
    return output

cv2.imshow("Mode Filter", mode_filter(gray))

# GAUSSIAN NOISE

mean = 0
sigma = 25

gaussian_noise = np.random.normal(mean, sigma, im_cv.shape)

gaussian_noise = gaussian_noise.astype(np.uint8)

gaussian_noisy = cv2.add(im_cv, gaussian_noise)

cv2.imshow("Gaussian Noise", gaussian_noisy)

# Gaussian noise removal using Gaussian filter
gaussian_removed = cv2.GaussianBlur(gaussian_noisy, (5,5), 0)

cv2.imshow("Gaussian Noise Removed", gaussian_removed)

# IMAGE AVERAGING

images = []
for i in range(5):
    n = np.random.normal(0, 25, im_cv.shape).astype(np.uint8)
    images.append(cv2.add(im_cv, n).astype(np.float32))

avg_image = np.uint8(np.mean(images, axis=0))
cv2.imshow("Image Averaging Result", avg_image)

# PERIODIC NOISE

rows, cols = gray.shape

x = np.arange(cols)
y = np.arange(rows)

X, Y = np.meshgrid(x, y)

sin_noise = 50 * np.sin(2 * np.pi * X / 30)

periodic_noise = gray + sin_noise

periodic_noise = np.uint8(np.clip(periodic_noise, 0, 255))

cv2.imshow("Periodic Noise", periodic_noise)

# Remove periodic noise using Average Filter
periodic_removed = cv2.blur(periodic_noise, (5,5))

cv2.imshow("Periodic Noise Removed", periodic_removed)

# SALT & PEPPER


image = cv2.imread("image.jpg", 0)

prob = 0.004

output = image.copy()

rand = np.random.rand(*image.shape)

# Pepper noise
output[rand < prob] = 0

# Salt noise
output[rand > 1 - prob] = 255

cv2.imshow("Salt and Pepper Noise", output)

# Remove Salt & Pepper using Median Filter
salt_removed = cv2.medianBlur(output, 5)

cv2.imshow("Salt and Pepper Removed", salt_removed)

# OUTLIER METHOD

noisy_img = cv2.imread("Noisy.jpg", 0)

kernel = np.array([
    [1/8, 1/8, 1/8],
    [1/8,   0, 1/8],
    [1/8, 1/8, 1/8]
])

# Apply averaging filter
average = cv2.filter2D(noisy_img, -1, kernel)

# Difference between noisy image and average
diff = np.abs(noisy_img - average)

# Threshold
threshold = 40

# Output image
output = noisy_img.copy()

# Replace outlier pixels
output[diff > threshold] = average[diff > threshold]

cv2.imshow("Noisy Image", noisy_img)
cv2.imshow("Filtered Image", output)

# RGB → HSI

rgb_img = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)

R = rgb_img[:,:,0] / 255.0
G = rgb_img[:,:,1] / 255.0
B = rgb_img[:,:,2] / 255.0

num = 0.5*((R-G)+(R-B))
den = np.sqrt((R-G)**2 + (R-B)*(G-B)) + 1e-6

theta = np.arccos(num/den)

H = np.where(B<=G, theta, 2*np.pi-theta)/(2*np.pi)
S = 1 - (3/(R+G+B+1e-6))*np.minimum(np.minimum(R,G),B)
I = (R+G+B)/3

print("RGB to HSI Conversion Done")

# HSI → RGB (HSV)

cv2.imshow("HSI to RGB", cv2.cvtColor(im_cv, cv2.COLOR_BGR2HSV))

# MORPHOLOGY

img = cv2.imread("img.png", 0)
kernel = np.ones((3,3), np.uint8)

erosion = cv2.erode(img, kernel)
dilation = cv2.dilate(img, kernel)

cv2.imshow("Erosion Result", erosion)
cv2.imshow("Dilation Result", dilation)

img2 = cv2.imread("img2.png", 0)
kernel = np.ones((5,5), np.uint8)

erosion_open = cv2.erode(img2, kernel)
img_open = cv2.dilate(erosion_open, kernel)
cv2.imshow("Opening Result", img_open)

dilation_close = cv2.dilate(img2, kernel)
img_close = cv2.erode(dilation_close, kernel)
cv2.imshow("Closing Result", img_close)

cv2.imshow("Internal Boundary", img - erosion)
cv2.imshow("External Boundary", dilation - img)
cv2.imshow("Morphological Gradient", dilation - erosion)



# SOBEL EDGE DETECTOR


img = cv2.imread("img.png", 0)

sobel_kernelx = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

sobel_kernely = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])

sobelx = cv2.filter2D(img, cv2.CV_64F, sobel_kernelx)
sobely = cv2.filter2D(img, cv2.CV_64F, sobel_kernely)

# Gradient Magnitude
sobel = cv2.magnitude(sobelx, sobely)

# Convert to 8-bit
sobel = cv2.convertScaleAbs(sobel)

cv2.imshow("Sobel Edge Detector", sobel)

# PREWITT EDGE DETECTOR

prewitt_kernelx = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

prewitt_kernely = np.array([
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
])

prewittx = cv2.filter2D(img, cv2.CV_64F, prewitt_kernelx)
prewitty = cv2.filter2D(img, cv2.CV_64F, prewitt_kernely)

# Gradient Magnitude
prewitt = cv2.magnitude(prewittx, prewitty)

# Convert to 8-bit
prewitt = cv2.convertScaleAbs(prewitt)

cv2.imshow("Prewitt Edge Detector", prewitt)

# ROBERT EDGE DETECTOR

robert_kernelx = np.array([
    [1, 0],
    [0, -1]
])

robert_kernely = np.array([
    [0, 1],
    [-1, 0]
])

robertx = cv2.filter2D(img, cv2.CV_64F, robert_kernelx)
roberty = cv2.filter2D(img, cv2.CV_64F, robert_kernely)

# Gradient Magnitude
robert = cv2.magnitude(robertx, roberty)

# Convert to 8-bit
robert = cv2.convertScaleAbs(robert)

cv2.imshow("Robert Edge Detector", robert)

# LAPLACIAN FILTER

img = cv2.imread("img.png", 0)

# Laplacian Kernel
laplacian_kernel = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
])

# Apply convolution
laplacian = cv2.filter2D(img, cv2.CV_64F, laplacian_kernel)

# Convert to 8-bit image
laplacian = cv2.convertScaleAbs(laplacian)

# Display result
cv2.imshow("Laplacian Filter", laplacian)




cv2.waitKey(0)
cv2.destroyAllWindows()