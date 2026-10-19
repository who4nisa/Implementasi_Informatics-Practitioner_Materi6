import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("==================================================================")
print(" STUDI KASUS SKRIPSI: KLASIFIKASI PENYEBAB DEFORESTASI DENGAN CNN")
print(" Topik       : Deep Learning & Computer Vision (Citra Satelit)")
print(" Nama File   : implementasi_materi6a.py")
print(" Kandidat    : Anisa Julianti (NPM: G1A023052)")
print(" Universitas : Universitas Bengkulu")
print("==================================================================\n")

# ==================================================================
# 1. GENERATOR DATASET SINTETIS BERBASIS SPATIAL TEXTURE (PERBAIKAN DIMENSI)
# ==================================================================
print("--- [Langkah 1] Membangkitkan Dataset Simulasi dengan Pola Spasial ---")
np.random.seed(42)
tf.random.set_seed(42)

n_samples_per_class = 200  # Total 800 sampel data citra
img_height, img_width = 128, 128
n_channels = 3  

X_list = []
y_list = []

for class_id in range(4):
    for _ in range(n_samples_per_class):
        img = np.random.rand(img_height, img_width, n_channels) * 0.2
        
        if class_id == 0:  # Perkebunan Ilegal
            img[:, :, 1] += np.random.uniform(0.5, 0.8, size=(img_height, img_width))
            img[30:90, 30:90, :] += 0.2  
        elif class_id == 1:  # Pertanian Rakyat (Diperbaiki menggunakan step 4 agar ukurannya presisi)
            sub_h = len(np.arange(0, img_height, 4))
            img[::4, :, :] += np.random.uniform(0.4, 0.7, size=(sub_h, img_width, n_channels))
        elif class_id == 2:  # Kebakaran Hutan / Karhutla
            img[:, :, 0] += np.random.uniform(0.6, 0.9, size=(img_height, img_width))
            img[40:80, 40:80, :] *= 0.3  
        elif class_id == 3:  # Tambang Terbuka
            img += np.random.uniform(0.5, 0.8, size=(img_height, img_width, n_channels))
            img[20:100, 20:100, 0] += 0.3
            
        img = np.clip(img, 0.0, 1.0)
        X_list.append(img)
        y_list.append(class_id)

X_data = np.array(X_list, dtype=np.float32)
y_data = np.array(y_list, dtype=np.int32)

print(f"Total Sampel Citra Terbentuk : {X_data.shape[0]} data")
print(f"Dimensi Tensor Citra         : {X_data.shape}")
print(f"Distribusi Kelas Target      : {np.bincount(y_data)}\n")

# Pembagian Dataset: Training (70%), Validation (15%), Testing (15%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_data, y_data, test_size=0.15, random_state=42, stratify=y_data
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1764, random_state=42, stratify=y_train_full
)

print(f"Jumlah Data Latih (Training)     : {len(X_train)} sampel")
print(f"Jumlah Data Validasi (Validation): {len(X_val)} sampel")
print(f"Jumlah Data Uji (Testing)        : {len(X_test)} sampel\n")

# ==================================================================
# 2. ARSITEKTUR CNN TINGKAT LANJUT
# ==================================================================
print("--- [Langkah 2] Membangun Arsitektur CNN Tingkat Lanjut ---")

model = models.Sequential([
    layers.Input(shape=(img_height, img_width, n_channels)),
    
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==================================================================
# 3. PELATIHAN MODEL
# ==================================================================
print("--- [Langkah 3] Menjalankan Proses Pelatihan Model (Training) ---")

cb_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=cb_list,
    verbose=1
)
print("\n[Sukses] Pelatihan model selesai.\n")

# ==================================================================
# 4. EVALUASI MODEL
# ==================================================================
print("--- [Langkah 4] Evaluasi Performa pada Data Uji ---")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Akurasi Model pada Data Uji : {test_accuracy * 100:.2f}%")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = ["Perkebunan Ilegal", "Pertanian Rakyat", "Kebakaran Hutan", "Tambang Terbuka"]
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))