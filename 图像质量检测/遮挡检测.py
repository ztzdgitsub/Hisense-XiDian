import cv2
import numpy as np
import os


def detect_black_occlusion(
        img,
        black_thresh=40,
        area_ratio_thresh=0.15,
        std_thresh=15.0,
        select_mode="largest"   # "smoothest" 或 "largest"
):
    """
    检测大面积黑色遮挡块（支持按最大面积或最光滑边界选择遮挡块）
    返回: (is_occluded, occlusion_ratio) 元组
    """

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. 阈值分割黑色区域
    _, mask = cv2.threshold(gray, black_thresh, 255, cv2.THRESH_BINARY_INV)

    # 2. 连通区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False, 0.0  # 返回 (未遮挡, 遮挡比例0)

    # ===== 连通域选择策略 =====
    chosen_contour = None

    if select_mode == "largest":
        chosen_contour = max(contours, key=cv2.contourArea)

    elif select_mode == "smoothest":
        smoothness_scores = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 1:
                continue
            perimeter = cv2.arcLength(cnt, True)
            smoothness = (perimeter * perimeter) / area
            smoothness_scores.append((smoothness, cnt))

        if len(smoothness_scores) == 0:
            return False, 0.0

        chosen_contour = min(smoothness_scores, key=lambda x: x[0])[1]

    else:
        return False, 0.0

    cnt = chosen_contour
    area = cv2.contourArea(cnt)
    area_ratio = area / (h * w)

    # 3. 面积过滤
    if area_ratio < area_ratio_thresh:
        return False, area_ratio

    # 4. 遮挡区域纹理检测
    mask_region = np.zeros_like(gray)
    cv2.drawContours(mask_region, [cnt], -1, 255, -1)
    block_pixels = gray[mask_region == 255]
    block_std = np.std(block_pixels)

    if block_std > std_thresh:
        return False, area_ratio

    # ===== 输出返回布尔值和遮挡比例 =====
    return True, area_ratio


# ============================================================
# 按 is_occluded() 输出格式修改 remove_occluded_images
# ============================================================
def remove_occluded_images(input_dir, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_images = 0
    total_G_images = 0
    occluded_non_G = 0
    occluded_G = 0

    for fn in os.listdir(input_dir):
        path = os.path.join(input_dir, fn)
        img = cv2.imread(path)
        if img is None:
            continue

        total_images += 1

        # 是否是 _G 图片
        name_without_ext, ext = os.path.splitext(fn)
        is_G_image = name_without_ext.endswith('_G')
        if is_G_image:
            total_G_images += 1

        # 使用你的 detect_black_occlusion 函数
        is_occ, occlusion_ratio = detect_black_occlusion(img)

        if is_occ:
            print(f"图片{fn}存在遮挡，遮挡面积为{occlusion_ratio:.2%}")
            if is_G_image:
                occluded_G += 1
            else:
                occluded_non_G += 1
        else:
            print(f"图片{fn}未被遮挡")
            cv2.imwrite(os.path.join(output_dir, fn), img)

    # ===== 输出格式完全对齐 is_occluded() 的版本 =====
    print(f"\n处理完成:")
    print(f"  - 输入图片总数: {total_images}")
    print(f"  - 保留图片数: {total_images - occluded_non_G - occluded_G}")


# ============================================================
if __name__ == "__main__":
    input_dir = "E:/archive/G2"
    output_dir = "E:/archive/R1"
    remove_occluded_images(input_dir, output_dir)
