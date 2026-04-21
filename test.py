from ultralytics import YOLO
import cv2

# 加载你训练好的模型
model = YOLO('runs/detect/fire_smoke_optimized1/weights/best.pt')

# 测试单张图片
results = model('test1.png', device='cpu')  # 使用CPU推理，避免CUDA兼容性问题
results[0].show()  # 显示结果

# 测试摄像头实时检测
# cap = cv2.VideoCapture(0)  # 0为默认摄像头
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     results = model(frame)
#     annotated = results[0].plot()
#     cv2.imshow('Fire & Smoke Detection', annotated)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()