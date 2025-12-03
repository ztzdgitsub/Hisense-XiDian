from ultralytics import YOLO

def main():
    # 加载预训练 YOLOv11n 权重
    model = YOLO("yolo11n.pt")  # 或你的本地路径
    # model = YOLO("/media/disk_new/WHB/xingren/12_1080p_test/model/yolov11n_custom/weights/best.pt")

    # 开始训练
    model.train(
        data="/media/disk_new/WHB/xingren/12_1080p_test/dataset/1129/data.yaml",      # 数据集配置
        epochs=300,            # 训练轮数（根据你GPU调整）
        imgsz=640,             # 输入图片尺寸
        batch=16,              # 批大小
        device=0,              # GPU ID
        optimizer="Adam",      # 优化器，可选 SGD/Adam/AdamW
        cos_lr=True,           # 余弦学习率调度
        mosaic=1.0,            # 多尺度增强
        hsv_h=0.015,           # 色彩增强
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,         # 平移
        scale=0.5,             # 随机缩放
        fliplr=0.5,            # 左右翻转
        patience=0,           # EarlyStopping

        project="/media/disk_new/WHB/xingren/12_1080p_test/model",   # 自定义保存目录
        name="yolov11n_custom_1201",    # 保存子目录名
        exist_ok=True,             # 若已有同名目录允许覆盖
        save_period=10,            # 可选：每 10 epoch 保存一次

    )

    print("训练完成！")

if __name__ == "__main__":
    main()
