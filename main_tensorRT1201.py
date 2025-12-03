import cv2
import threading
import queue
import os
import time
from ultralytics import YOLO

VIDEO_FILE = "test1.mp4"
NUM_STREAMS = 4          # 先别直接 12，先试 4 看上限
ENGINE_FILE = "best.engine"

if not os.path.exists(VIDEO_FILE):
    print(f"错误: 视频文件未找到: {VIDEO_FILE}")
    exit()

if not os.path.exists(ENGINE_FILE):
    print(f"错误: TensorRT引擎文件未找到: {ENGINE_FILE}")
    exit()

frame_queue = queue.Queue(maxsize=NUM_STREAMS * 5)
stop_event = threading.Event()


def process_stream(stream_index, video_path):
    """每个线程自己加载一个 YOLO(engine)，只做推理，不做可视化"""
    print(f"[Stream {stream_index}] 正在加载引擎...")
    model = YOLO(ENGINE_FILE)
    print(f"[Stream {stream_index}] 引擎加载完成")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Stream {stream_index}] 无法打开视频 {video_path}")
        return

    frame_counter = 0
    start_time = time.time()
    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 纯推理
            results = model(frame, verbose=False)

            # 把 原始帧 + 结果 交给主线程画
            try:
                frame_queue.put((stream_index, frame, results[0]), timeout=1)
            except queue.Full:
                pass

            frame_counter += 1
            if time.time() - start_time > 1:
                fps = frame_counter / (time.time() - start_time)
                print(f"[Stream {stream_index}] 推理 FPS: {fps:.2f}")
                frame_counter = 0
                start_time = time.time()
    finally:
        cap.release()
        print(f"[Stream {stream_index}] 已停止")


if __name__ == "__main__":
    # 创建显示窗口
    for i in range(NUM_STREAMS):
        cv2.namedWindow(f"Stream {i}", cv2.WINDOW_NORMAL)

    threads = []
    for i in range(NUM_STREAMS):
        t = threading.Thread(target=process_stream, args=(i, VIDEO_FILE))
        t.start()
        threads.append(t)

    total_frames = 0
    t0 = time.time()
    try:
        while True:
            try:
                stream_index, frame, result = frame_queue.get(timeout=1)
            except queue.Empty:
                if not any(t.is_alive() for t in threads):
                    break
                continue

            annotated = result.plot(img=frame)
            total_frames += 1

            cv2.imshow(f"Stream {stream_index}", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    finally:
        for t in threads:
            t.join()
        cv2.destroyAllWindows()
        dt = time.time() - t0
        if dt > 0:
            print(f"\n总帧数: {total_frames}, 总时间: {dt:.2f}s, 平均显示 FPS: {total_frames/dt:.2f}")
