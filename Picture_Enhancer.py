import cv2
import time

# Start total timer
start_total = time.time()

# 1️⃣ Create Super Resolution object
sr = cv2.dnn_superres.DnnSuperResImpl_create()
print(f"[{round(time.time() - start_total, 2)}s] Super Resolution object created")

# 2️⃣ Load EDSR model
sr.readModel("EDSR_x4.pb")
sr.setModel("edsr", 4)  # 4x upscaling
print(f"[{round(time.time() - start_total, 2)}s] EDSR model loaded")

# 3️⃣ Read input image
img = cv2.imread("Data Science.jpg")
print(f"[{round(time.time() - start_total, 2)}s] Image loaded ({img.shape[1]}x{img.shape[0]})")

# 4️⃣ Upscale image
print(f"[{round(time.time() - start_total, 2)}s] Starting upscaling...")
upscaled = sr.upsample(img)
print(f"[{round(time.time() - start_total, 2)}s] Upscaling done")

# 5️⃣ Save output
cv2.imwrite("DS_hd.jpg", upscaled)
print(f"[{round(time.time() - start_total, 2)}s] Image saved as DS_hd.jpg")

# 6️⃣ Total processing time
print(f"Total processing time: {round(time.time() - start_total, 2)} seconds")
