# import cv2
# import numpy as np
# import os
# import time
# from pathlib import Path
# python
# def fourier_blur_detect(img_gray, threshold=0.015):
#     """基于高频能量比例的傅里叶模糊检测"""
#     dft = np.fft.fft2(img_gray)
#     dft_shift = np.fft.fftshift(dft)
#
#     rows, cols = img_gray.shape
#     crow, ccol = rows // 2, cols // 2
#
#     # 用中心低频区域构建 mask
#     mask = np.zeros((rows, cols), np.uint8)
#     mask[crow-30:crow+30, ccol-30:ccol+30] = 1
#
#     # 高频成分：整体频谱 - 低频区域
#     high_freq = dft_shift * (1 - mask)
#
#     # 高频能量占比
#     energy_ratio = np.sum(np.abs(high_freq) ** 2) / np.sum(np.abs(dft_shift) ** 2)
#
#     # 高频能量低 => 模糊
#     return energy_ratio < threshold, energy_ratio
#
#
# if __name__ == "__main__":
#
#     root = "/home/amax/XD/Image Blur/datasets/GOPRO_Large/train/GOPR0372_07_00/"
#     threshold = 0.020
#
#     # ====== 指标统计 ======
#     TP = TN = FP = FN = 0
#     total_images = 0
#     total_time = 0
#
#     # 支持所有模糊文件夹：blur / blur_gamma / blur_xxx
#     blurry_prefix = ["blur"]
#
#     # 遍历所有图片
#     for imagePath in Path(root).rglob("*.*"):
#         if imagePath.suffix.lower() not in [".jpg", ".png", ".jpeg", ".bmp"]:
#             continue
#
#         parent = imagePath.parent.name
#
#         # 读取图片
#         img = cv2.imread(str(imagePath), cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             print(f"Cannot read: {imagePath}")
#             continue
#
#         start = time.time()
#         pred_blur, ratio = fourier_blur_detect(img, threshold)
#         total_time += (time.time() - start)
#         total_images += 1
#
#         pred = "Blurry" if pred_blur else "Not Blurry"
#
#         # GT 判断（文件夹名）
#         if any(parent.startswith(p) for p in blurry_prefix):
#             gt = "Blurry"
#         else:
#             gt = "Not Blurry"
#
#         # === 统计混淆矩阵 ===
#         if gt == "Blurry" and pred == "Blurry":
#             TP += 1
#         elif gt == "Not Blurry" and pred == "Not Blurry":
#             TN += 1
#         elif gt == "Not Blurry" and pred == "Blurry":
#             FP += 1
#         else:  # gt Blurry but pred Not Blurry
#             FN += 1
#
#         print(f"{imagePath.name}: GT={gt}, Pred={pred}, Ratio={ratio:.4f}")
#
#     # ====== 计算最终指标 ======
#     accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
#     recall = TP / (TP + FN + 1e-6)       # 检出率
#     precision = TP / (TP + FP + 1e-6)
#     f1 = 2 * precision * recall / (precision + recall + 1e-6)
#     avg_time = total_time / (total_images + 1e-6)
#
#     print("\n============ Evaluation Result ============")
#     print(f"Total Images    : {total_images}")
#     print(f"Accuracy        : {accuracy:.4f}")
#     print(f"Recall(检出率)   : {recall:.4f}")
#     print(f"Precision       : {precision:.4f}")
#     print(f"F1-score        : {f1:.4f}")
#     print(f"Avg Time/Image  : {avg_time * 1000:.2f} ms")
#     print("===========================================\n")
#

import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

#pytorch

def fourier_blur_detect_torch(img_tensor, threshold=0.02):
    """
    输入: img_tensor shape = (1, H, W), uint8 / float32
    返回: pred_blur(bool), energy_ratio(float)
    """

    img = img_tensor.float()


    dft = torch.fft.fft2(img)
    dft_shift = torch.fft.fftshift(dft)

    H, W = img.shape[-2:]
    crow, ccol = H // 2, W // 2


    mask = torch.ones((H, W), device=img.device)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0


    high_freq = dft_shift * mask


    energy_ratio = (torch.sum(torch.abs(high_freq) ** 2) /
                    torch.sum(torch.abs(dft_shift) ** 2) + 1e-6).item()


    pred_blur = energy_ratio < threshold

    return pred_blur, energy_ratio



class FourierBlurDataset(Dataset):
    def __init__(self, root):
        self.image_paths = []
        for p in Path(root).rglob("*.*"):
            if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                self.image_paths.append(str(p))

        self.blurry_prefix = ["blur"]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]


        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


        parent = os.path.basename(os.path.dirname(path))
        gt = 1 if any(parent.startswith(p) for p in self.blurry_prefix) else 0

        return img_tensor, gt, path


def evaluate(model_loader, threshold=0.02):
    TP = TN = FP = FN = 0
    total_images = 0
    total_time = 0

    for img_tensor, gt, path in model_loader:
        img_tensor = img_tensor.squeeze(0)  # 去掉 batch 维度
        gt = gt.item()

        start = time.time()

        pred_blur, ratio = fourier_blur_detect_torch(img_tensor, threshold)
        total_time += time.time() - start
        total_images += 1

        pred = 1 if pred_blur else 0


        if gt == 1 and pred == 1:
            TP += 1
        elif gt == 0 and pred == 0:
            TN += 1
        elif gt == 0 and pred == 1:
            FP += 1
        else:
            FN += 1

        print(f"{os.path.basename(path[0])}: GT={gt}, Pred={pred}, Ratio={ratio:.4f}")


    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    avg_time = total_time / (total_images + 1e-6)

    print("\n============ Evaluation Result ============")
    print(f"Total Images    : {total_images}")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Recall(检出率)   : {recall:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"F1-score        : {f1:.4f}")
    print(f"Avg Time/Image  : {avg_time * 1000:.2f} ms")
    print("===========================================\n")



if __name__ == "__main__":
    root = "/home/amax/XD/Image Blur/datasets/GOPRO_Large/test/GOPR0869_11_00/"
    threshold = 0.018

    dataset = FourierBlurDataset(root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    evaluate(loader, threshold)
