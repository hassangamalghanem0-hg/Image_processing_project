from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from scipy import stats
import base64
import io
import os
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED

def file_to_b64(img):
    _, buf = cv2.imencode('.png', img)
    return base64.b64encode(buf).decode('utf-8')

def read_image(file):
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def read_gray(file):
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

# ── 1. IMAGE INFO ──────────────────────────────
@app.route('/api/info', methods=['POST'])
def info():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    if img is None: return jsonify(error='Cannot decode image'), 400
    h, w, c = img.shape
    return jsonify(
        height=h, width=w, channels=c,
        preview=file_to_b64(img),
        message=f"{w}×{h} px · {c} channels"
    )

# ── 2. COMPLEMENT ─────────────────────────────
@app.route('/api/complement', methods=['POST'])
def complement():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    result = 255 - img
    return jsonify(original=file_to_b64(img), result=file_to_b64(result))

# ── 3. RGB CHANNELS ───────────────────────────
@app.route('/api/channels', methods=['POST'])
def channels():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]
    return jsonify(
        original=file_to_b64(img),
        blue=file_to_b64(b),
        green=file_to_b64(g),
        red=file_to_b64(r)
    )

# ── 4. COLOR OPERATIONS ────────────────────────
@app.route('/api/color_ops', methods=['POST'])
def color_ops():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)

    red_boost = img.copy()
    red_boost[:, :, 2] = cv2.add(red_boost[:, :, 2], 50)

    swap_rg = img.copy()
    swap_rg[:, :, 1] = img[:, :, 2]
    swap_rg[:, :, 2] = img[:, :, 1]

    no_red = img.copy()
    no_red[:, :, 2] = 0

    return jsonify(
        original=file_to_b64(img),
        red_boost=file_to_b64(red_boost),
        swap_rg=file_to_b64(swap_rg),
        no_red=file_to_b64(no_red)
    )

# ── 5. ROI ────────────────────────────────────
@app.route('/api/roi', methods=['POST'])
def roi():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    x = int(request.form.get('x', 100))
    y = int(request.form.get('y', 100))
    w = int(request.form.get('w', 200))
    h = int(request.form.get('h', 200))
    H, W = img.shape[:2]
    x = min(x, W - 1); y = min(y, H - 1)
    w = min(w, W - x); h = min(h, H - y)
    region = img[y:y+h, x:x+w]
    viz = img.copy()
    cv2.rectangle(viz, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return jsonify(original=file_to_b64(viz), result=file_to_b64(region))

# ── 6. ARITHMETIC & LOGICAL OPS ───────────────
@app.route('/api/arithmetic', methods=['POST'])
def arithmetic():
    f1 = request.files.get('image')
    f2 = request.files.get('image2')
    if not f1 or not f2: return jsonify(error='Two images required'), 400
    img1 = read_image(f1)
    img2 = read_image(f2)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return jsonify(
        original=file_to_b64(img1),
        image2=file_to_b64(img2),
        add=file_to_b64(cv2.add(img1, img2)),
        subtract=file_to_b64(cv2.subtract(img1, img2)),
        multiply=file_to_b64(cv2.multiply(img1, img2)),
        divide=file_to_b64(cv2.divide(img1, img2)),
        bitwise_and=file_to_b64(cv2.bitwise_and(img1, img2)),
        bitwise_or=file_to_b64(cv2.bitwise_or(img1, img2)),
        bitwise_xor=file_to_b64(cv2.bitwise_xor(img1, img2))
    )

# ── 7. GRAYSCALE + HISTOGRAM ──────────────────
@app.route('/api/histogram', methods=['POST'])
def histogram():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()
    return jsonify(
        original=file_to_b64(img),
        gray=file_to_b64(gray),
        histogram=hist
    )

# ── 8. GAMMA CORRECTION ───────────────────────
@app.route('/api/gamma', methods=['POST'])
def gamma():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gamma_val = float(request.form.get('gamma', 2.0))
    im_norm = gray / 255.0
    corrected = np.uint8((im_norm ** gamma_val) * 255)
    return jsonify(
        original=file_to_b64(gray),
        result=file_to_b64(corrected),
        gamma=gamma_val
    )

# ── 9. HISTOGRAM STRETCHING + EQUALIZATION ─────
@app.route('/api/hist_enhance', methods=['POST'])
def hist_enhance():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mn, mx = int(np.min(gray)), int(np.max(gray))
    stretch = np.uint8(((gray.astype(np.float32) - mn) / (mx - mn + 1e-6)) * 255)
    equalized = cv2.equalizeHist(gray)
    orig_hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()
    str_hist  = cv2.calcHist([stretch], [0], None, [256], [0, 256]).flatten().tolist()
    eq_hist   = cv2.calcHist([equalized], [0], None, [256], [0, 256]).flatten().tolist()
    return jsonify(
        original=file_to_b64(gray),
        stretched=file_to_b64(stretch),
        equalized=file_to_b64(equalized),
        orig_hist=orig_hist,
        str_hist=str_hist,
        eq_hist=eq_hist,
        min_val=mn, max_val=mx
    )

# ── 10. FILTERS ────────────────────────────────
@app.route('/api/filters', methods=['POST'])
def filters():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    average = cv2.blur(img, (5, 5))
    median  = cv2.medianBlur(img, 5)
    rank_order = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    max_f  = cv2.dilate(gray, np.ones((3, 3), np.uint8))
    min_f  = cv2.erode(gray, np.ones((3, 3), np.uint8))

    def mode_filter(im):
        h, w = im.shape
        out = np.zeros_like(im)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                win = im[i-1:i+2, j-1:j+2].flatten()
                out[i, j] = stats.mode(win, keepdims=True)[0]
        return out

    small = cv2.resize(gray, (80, 80))
    mode_small = mode_filter(small)
    mode_full  = cv2.resize(mode_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    return jsonify(
        original=file_to_b64(img),
        average=file_to_b64(average),
        median=file_to_b64(median),
        rank_order=file_to_b64(rank_order),
        max_filter=file_to_b64(max_f),
        min_filter=file_to_b64(min_f),
        mode_filter=file_to_b64(mode_full)
    )

# ── 11. GAUSSIAN NOISE ─────────────────────────
@app.route('/api/gaussian_noise', methods=['POST'])
def gaussian_noise():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    sigma = float(request.form.get('sigma', 25))
    noise = np.random.normal(0, sigma, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    removed = cv2.GaussianBlur(noisy, (5, 5), 0)
    return jsonify(
        original=file_to_b64(img),
        noisy=file_to_b64(noisy),
        removed=file_to_b64(removed),
        sigma=sigma
    )

# ── 12. IMAGE AVERAGING ────────────────────────
@app.route('/api/averaging', methods=['POST'])
def averaging():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    images = []
    for _ in range(5):
        n = np.random.normal(0, 25, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
        images.append(noisy.astype(np.float32))
    avg = np.uint8(np.mean(images, axis=0))
    sample_noisy = np.clip(img.astype(np.int16) + np.random.normal(0, 25, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
    return jsonify(
        original=file_to_b64(img),
        noisy_sample=file_to_b64(sample_noisy),
        averaged=file_to_b64(avg)
    )

# ── 13. PERIODIC NOISE ─────────────────────────
@app.route('/api/periodic_noise', methods=['POST'])
def periodic_noise():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    freq = float(request.form.get('freq', 30))
    amp  = float(request.form.get('amp', 50))
    sin_noise = amp * np.sin(2 * np.pi * X / freq)
    periodic = np.uint8(np.clip(gray.astype(np.float32) + sin_noise, 0, 255))
    removed  = cv2.blur(periodic, (5, 5))
    return jsonify(
        original=file_to_b64(gray),
        noisy=file_to_b64(periodic),
        removed=file_to_b64(removed),
        freq=freq, amp=amp
    )

# ── 14. SALT & PEPPER ─────────────────────────
@app.route('/api/salt_pepper', methods=['POST'])
def salt_pepper():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prob = float(request.form.get('prob', 0.004))
    out  = gray.copy()
    rnd  = np.random.rand(*gray.shape)
    out[rnd < prob] = 0
    out[rnd > 1 - prob] = 255
    removed = cv2.medianBlur(out, 5)
    return jsonify(
        original=file_to_b64(gray),
        noisy=file_to_b64(out),
        removed=file_to_b64(removed),
        prob=prob
    )

# ── 15. OUTLIER METHOD ────────────────────────
@app.route('/api/outlier', methods=['POST'])
def outlier():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[1/8,1/8,1/8],[1/8,0,1/8],[1/8,1/8,1/8]])
    average = cv2.filter2D(gray, -1, kernel)
    diff = np.abs(gray.astype(np.int16) - average.astype(np.int16)).astype(np.uint8)
    threshold = int(request.form.get('threshold', 40))
    out = gray.copy()
    out[diff > threshold] = average[diff > threshold]
    return jsonify(
        original=file_to_b64(gray),
        diff=file_to_b64(diff),
        result=file_to_b64(out),
        threshold=threshold
    )

# ── 16. RGB → HSI ─────────────────────────────
@app.route('/api/hsi', methods=['POST'])
def hsi():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R = rgb[:, :, 0] / 255.0
    G = rgb[:, :, 1] / 255.0
    B = rgb[:, :, 2] / 255.0
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B)) + 1e-6
    theta = np.arccos(np.clip(num / den, -1, 1))
    H = np.where(B <= G, theta, 2 * np.pi - theta) / (2 * np.pi)
    S = 1 - (3 / (R + G + B + 1e-6)) * np.minimum(np.minimum(R, G), B)
    I = (R + G + B) / 3
    H_img = np.uint8(H * 255)
    S_img = np.uint8(np.clip(S, 0, 1) * 255)
    I_img = np.uint8(np.clip(I, 0, 1) * 255)
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return jsonify(
        original=file_to_b64(img),
        hue=file_to_b64(H_img),
        saturation=file_to_b64(S_img),
        intensity=file_to_b64(I_img),
        hsv=file_to_b64(hsv)
    )

# ── 17. MORPHOLOGY ────────────────────────────
@app.route('/api/morphology', methods=['POST'])
def morphology():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    erosion   = cv2.erode(binary, k3)
    dilation  = cv2.dilate(binary, k3)
    opening   = cv2.dilate(cv2.erode(binary, k5), k5)
    closing   = cv2.erode(cv2.dilate(binary, k5), k5)
    int_bound = cv2.subtract(binary, erosion)
    ext_bound = cv2.subtract(dilation, binary)
    gradient  = cv2.subtract(dilation, erosion)
    return jsonify(
        original=file_to_b64(binary),
        erosion=file_to_b64(erosion),
        dilation=file_to_b64(dilation),
        opening=file_to_b64(opening),
        closing=file_to_b64(closing),
        internal=file_to_b64(int_bound),
        external=file_to_b64(ext_bound),
        gradient=file_to_b64(gradient)
    )

# ── 18. EDGE DETECTION ────────────────────────
@app.route('/api/edges', methods=['POST'])
def edges():
    f = request.files.get('image')
    if not f: return jsonify(error='No image'), 400
    img = read_image(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    sobel_ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sx = cv2.filter2D(gray, cv2.CV_64F, sobel_kx)
    sy = cv2.filter2D(gray, cv2.CV_64F, sobel_ky)
    sobel = cv2.convertScaleAbs(cv2.magnitude(sx, sy))

    prewitt_kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prewitt_ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    px = cv2.filter2D(gray, cv2.CV_64F, prewitt_kx)
    py = cv2.filter2D(gray, cv2.CV_64F, prewitt_ky)
    prewitt = cv2.convertScaleAbs(cv2.magnitude(px, py))

    robert_kx = np.array([[1,0],[0,-1]])
    robert_ky = np.array([[0,1],[-1,0]])
    rx = cv2.filter2D(gray, cv2.CV_64F, robert_kx)
    ry = cv2.filter2D(gray, cv2.CV_64F, robert_ky)
    robert = cv2.convertScaleAbs(cv2.magnitude(rx, ry))

    lap_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    laplacian  = cv2.convertScaleAbs(cv2.filter2D(gray, cv2.CV_64F, lap_kernel))

    return jsonify(
        original=file_to_b64(gray),
        sobel=file_to_b64(sobel),
        prewitt=file_to_b64(prewitt),
        robert=file_to_b64(robert),
        laplacian=file_to_b64(laplacian)
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)