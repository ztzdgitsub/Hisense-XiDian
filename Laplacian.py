# from imutils import paths
# import argparse
# import cv2
# import os
# import time
#
#python
# def variance_of_laplacian(image):
#     """计算图像 Laplacian 方差，用于判断是否模糊"""
#     return cv2.Laplacian(image, cv2.CV_64F).var()
#
#
# if __name__ == '__main__':
#     # 设置参数
#     ap = argparse.ArgumentParser()
#     ap.add_argument(
#         "-i",
#         "--images",
#         default="/home/amax/XD/Image Blur/datasets/GOPRO_Large/test/GOPR0869_11_00/",
#         help="Path to images root folder"
#     )
#     ap.add_argument("-t", "--threshold", type=float, default=100.0, help="Threshold")
#     args = vars(ap.parse_args())
#
#     # 评价指标计数器
#     TP = 0  # 真模糊判断为模糊
#     TN = 0  # 真清晰判断为清晰
#     FP = 0  # 清晰误报为模糊
#     FN = 0  # 模糊误判为清晰
#
#     total_time = 0
#     total_images = 0
#
#     # 遍历全部图片（包含 blur 和 sharp）
#     for imagePath in paths.list_images(args["images"]):
#         start_time = time.time()
#
#         image = cv2.imread(imagePath)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         # Laplacian 模糊得分
#         fm = variance_of_laplacian(gray)
#         pred = "Blurry" if fm < args["threshold"] else "Not Blurry"
#
#         # ====== Ground Truth（根据文件夹名称判断）======
#         blurry_folders = ["blur", "blur_gamma"]
#
#         folder = os.path.basename(os.path.dirname(imagePath))
#
#         if folder in blurry_folders:
#             gt = "Blurry"
#         else:
#             gt = "Not Blurry"
#
#         # ====== 统计指标 ======
#         if gt == "Blurry" and pred == "Blurry":
#             TP += 1
#         elif gt == "Not Blurry" and pred == "Not Blurry":
#             TN += 1
#         elif gt == "Not Blurry" and pred == "Blurry":
#             FP += 1
#         elif gt == "Blurry" and pred == "Not Blurry":
#             FN += 1
#
#         # 统计时间
#         total_time += (time.time() - start_time)
#         total_images += 1
#
#         # 输出
#         print(f"{imagePath}  ->  GT={gt}  Pred={pred}  Score={fm:.2f}")
#
#     # ====== 计算指标 ======
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     recall = TP / (TP + FN + 1e-6)
#     precision = TP / (TP + FP + 1e-6)
#     f1 = 2 * precision * recall / (precision + recall + 1e-6)
#
#     avg_time = total_time / total_images
#
#     print("\n======= Evaluation Result =======")
#     print(f"Total images     : {total_images}")
#     print(f"Accuracy         : {accuracy:.4f}")
#     print(f"Recall (检出率)   : {recall:.4f}")
#     print(f"Precision        : {precision:.4f}")
#     print(f"F1-score         : {f1:.4f}")
#     print(f"Avg time/image   : {avg_time * 1000:.2f} ms")
#     print("=================================\n")
#
#

import os
import cv2
import time
import torch
from torch.utils.data import Dataset, DataLoader
from imutils import paths

#pytorch

def variance_of_laplacian(image_tensor):
    """
    PyTorch 版本的 Laplacian 方差计算
    输入: (1, H, W) 的灰度图 tensor，值范围 0~255
    """
    # 转成 float32
    img = image_tensor.float()

    # 使用 PyTorch 的 Laplacian
    lap = torch.nn.functional.conv2d(
        img.unsqueeze(0),   # (1,1,H,W)
        weight=torch.tensor([[[[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]]]], dtype=torch.float32)
    )

    return lap.var().item()


# ===========================
# PyTorch Dataset
# ===========================
class BlurDataset(Dataset):
    def __init__(self, root):
        self.image_paths = list(paths.list_images(root))
        self.blurry_folders = ["blur", "blur_gamma"]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 转换为 (1, H, W) tensor
        gray_tensor = torch.from_numpy(gray).unsqueeze(0)

        # GT 标签
        folder = os.path.basename(os.path.dirname(path))

        gt = 1 if folder in self.blurry_folders else 0  # 1=blurry, 0=sharp

        return gray_tensor, gt, path


# ===========================
# Evaluation Loop
# ===========================
def evaluate(data_loader, threshold=100.0):
    TP = TN = FP = FN = 0
    total_time = 0
    total_images = 0

    for gray, gt, path in data_loader:
        gray = gray.squeeze(0)  # remove batch dim (1)

        start_time = time.time()

        # Laplacian 模糊得分 (PyTorch)
        fm = variance_of_laplacian(gray)

        pred = 1 if fm < threshold else 0  # 1=blurry

        # 分类统计
        if gt.item() == 1 and pred == 1:
            TP += 1
        elif gt.item() == 0 and pred == 0:
            TN += 1
        elif gt.item() == 0 and pred == 1:
            FP += 1
        elif gt.item() == 1 and pred == 0:
            FN += 1

        total_time += (time.time() - start_time)
        total_images += 1

        print(f"{path[0]}  GT={gt.item()}  Pred={pred}  Score={fm:.2f}")

    # ====== 指标 ======
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    avg_time = total_time / total_images

    print("\n======= Evaluation Result =======")
    print(f"Total images     : {total_images}")
    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Recall           : {recall:.4f}")
    print(f"Precision        : {precision:.4f}")
    print(f"F1-score         : {f1:.4f}")
    print(f"Avg time/image   : {avg_time * 1000:.2f} ms")
    print("=================================\n")


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    root = "/home/amax/XD/Image Blur/datasets/GOPRO_Large/train/GOPR0372_07_00/"

    dataset = BlurDataset(root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    evaluate(dataloader, threshold=100.0)
