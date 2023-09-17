import cv2 as cv
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from tensorflow import keras
import keras_preprocessing
img = cv.imread("E:\SatdatRapi\Test_2\DataTest242.png")

# resize citra dengan mengalikannya ukuran aslinya dengan 0.4
# contoh: 1920 x 2560 ==> 1920 x 0.4 = 768 ; 2560 x 0.4 = 1024 ==> hasilnya 768 x 1024
# img.shape[1] = kolom/lebar ; img.shape[0] = baris/tinggi
img = cv.resize(img, (300,100))

# konversi dari BGR ke grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert bgr to grayscale

# Normalisasi Cahaya
# citra kendaraan memiliki intensitas cahaya yang berbeda-beda maka normalkan terlebih dahulu
# cara menormalkan intensitas cahaya:
# 1. lakukan operasi opening di citra gray
# 2. lakukan pengurangan citra gray dengan citra hasil opening
# 3. citra hasil normalisasi bisa diubah ke citra BW (hitam putih) dengan pengambangan Otsu

# buat kernel dengan bentuk ellipse, diameter 20 piksel
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))

# Normalisasi cahaya (1)
# lakukan operasi opening ke citra grayscale dengan kernel yang sudah dibuat (var: kernel)
img_opening = cv.morphologyEx(img_gray, cv.MORPH_OPEN, kernel)

# Normalisasi cahaya (2)
# lakukan pengurangan citra grayscale dengan citra hasil opening
img_norm = img_gray - img_opening

# Normalisasi cahaya (3)
# konversi citra hasil normalisasi ke citra BW (hitam putih)
(thresh, img_norm_bw) = cv.threshold(img_norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# ==== Cek normalisasi START ====
# Untuk ngecek hasil sebelum dan sesudah dilakukan normalisasi
# Bisa di comment/uncomment

# buat citra bw tanpa normalisasi
(thresh, img_without_norm_bw) = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

fig = plt.figure(figsize=(10, 7))
row_fig = 2
column_fig = 2

fig.add_subplot(row_fig, column_fig, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("RGB")

fig.add_subplot(row_fig, column_fig, 2)
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.title("Grayscale")

fig.add_subplot(row_fig, column_fig, 3)
plt.imshow(img_without_norm_bw, cmap='gray')
plt.axis('off')
plt.title("Tanpa Normalisasi")

fig.add_subplot(row_fig, column_fig, 4)
plt.imshow(img_norm_bw, cmap='gray')
plt.axis('off')
plt.title("Dengan Normalisasi")

plt.show()