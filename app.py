"""
火灾烟雾检测 & 非法人员闯入检测前端展示界面
基于训练好的YOLOv8模型进行火灾、烟雾检测以及人员闯入检测
"""

import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import os
import tempfile
import time
import csv
import torch


def is_gpu_available():
    """检查GPU是否真正可用（严格的CUDA兼容性测试）"""
    try:
        if not torch.cuda.is_available():
            return False

        # 获取CUDA信息
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()

        if not cuda_version or device_count == 0:
            return False

        # 测试基本CUDA功能
        device = torch.device('cuda:0')

        # 测试1: 创建和操作张量（避免可能的不兼容操作）
        test_tensor = torch.tensor([1.0], device=device)
        _ = test_tensor + 1.0  # 简单操作

        # 测试2: 内存传输
        cpu_tensor = torch.tensor([1.0])
        gpu_tensor = cpu_tensor.to(device)
        back_to_cpu = gpu_tensor.cpu()

        # 测试3: 检查是否支持当前架构（通过尝试一个简单卷积）
        try:
            # 使用简单的卷积操作测试CUDA内核兼容性
            input_tensor = torch.randn(1, 3, 8, 8, device=device)
            conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1).to(device)
            output = conv(input_tensor)
            _ = output.cpu()  # 传输回CPU
        except Exception:
            return False

        # 清理缓存
        torch.cuda.empty_cache()

        return True

    except Exception:
        # CUDA不可用或兼容性问题
        return False


def get_available_device():
    """获取可用设备，优先GPU，失败时回退到CPU"""
    if is_gpu_available():
        return "cuda"
    else:
        return "cpu"


def load_training_metrics():
    """加载训练指标"""
    csv_path = "runs/detect/fire_smoke_optimized1/results.csv"
    metrics = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                # 提取关键指标
                metrics['precision'] = float(last_row.get('metrics/precision(B)', 0))
                metrics['recall'] = float(last_row.get('metrics/recall(B)', 0))
                metrics['mAP50'] = float(last_row.get('metrics/mAP50(B)', 0))
                metrics['mAP50_95'] = float(last_row.get('metrics/mAP50-95(B)', 0))
                metrics['epoch'] = int(last_row.get('epoch', 0))
    except Exception as e:
        st.warning(f"无法加载训练指标: {e}")
    return metrics


# 设置页面配置
st.set_page_config(
    page_title="火灾烟雾检测 & 人员闯入检测系统",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题和介绍
st.title("🔥 火灾烟雾检测 & 🚶 非法人员闯入检测系统")
st.markdown("""
    基于YOLOv8深度学习模型的智能检测系统。支持**火灾烟雾检测**和**非法人员闯入检测**两种模式。
    上传图片或使用摄像头进行实时检测。
""")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 设置")

    # 检测类型选择
    detection_type = st.radio(
        "检测类型",
        ["火灾烟雾检测", "非法闯入检测"],
        index=0,
        help="选择检测类型：火灾烟雾检测或非法人员闯入检测"
    )

    # 根据检测类型显示不同配置
    if detection_type == "火灾烟雾检测":
        # 模型选择（仅火灾烟雾检测需要）
        model_option = st.selectbox(
            "选择模型",
            ["最佳模型 (best.pt)", "最后模型 (last.pt)"],
            index=0
        )

        # 置信度阈值
        conf_threshold = st.slider(
            "置信度阈值",
            min_value=0.01,
            max_value=1.0,
            value=0.25,
            help="调整检测的置信度阈值，值越高误报越少但可能漏检"
        )
    else:
        # 人员检测置信度阈值
        conf_threshold = st.slider(
            "人员检测置信度阈值",
            min_value=0.01,
            max_value=1.0,
            value=0.5,
            help="调整人员检测的置信度阈值，值越高误报越少但可能漏检"
        )

        # 报警阈值
        alarm_threshold = st.slider(
            "报警触发人数",
            min_value=1,
            max_value=10,
            value=1,
            help="检测到多少人时触发闯入报警"
        )

    # 设备选择
    gpu_available = is_gpu_available()

    if gpu_available:
        device_options = ["自动选择", "CPU", "GPU"]
        default_index = 0  # 自动选择
        help_text = "选择模型推理使用的设备，GPU可加速推理"
    else:
        device_options = ["CPU"]
        default_index = 0  # 只有CPU选项
        help_text = "GPU不可用，将使用CPU进行推理"

    device_option = st.selectbox(
        "推理设备",
        device_options,
        index=default_index,
        help=help_text
    )

    # 检测模式（图片上传/摄像头）
    detection_mode = st.radio(
        "输入方式",
        ["图片上传", "摄像头实时检测"],
        index=0
    )

    st.divider()

    # 训练结果展示（仅火灾烟雾检测模式显示）
    if detection_type == "火灾烟雾检测":
        st.header("📊 训练结果")
        show_results = st.checkbox("显示训练结果图表", value=True)
        show_metrics = st.checkbox("显示关键指标", value=True)
        st.divider()

    # 关于
    st.header("ℹ️ 关于")
    if detection_type == "火灾烟雾检测":
        st.markdown("""
        - **模型**: YOLOv8s
        - **类别**: Fire, Smoke
        - **训练轮数**: 150
        - **数据集**: 自定义火灾烟雾数据集
        """)
    else:
        st.markdown("""
        - **模型**: YOLOv8n (预训练)
        - **检测目标**: 人员 (Person)
        - **数据集**: COCO 80类
        - **功能**: 非法闯入检测 & 报警
        """)

# ==================== 模型加载 ====================
@st.cache_resource
def load_fire_smoke_model(model_type="best"):
    """加载火灾烟雾检测模型"""
    if model_type == "best":
        model_path = "runs/detect/fire_smoke_optimized1/weights/best.pt"
    else:
        model_path = "runs/detect/fire_smoke_optimized1/weights/last.pt"

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"加载火灾烟雾模型失败: {e}")
        return None


@st.cache_resource
def load_person_detection_model():
    """加载人员检测模型（使用预训练的YOLOv8n）"""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"加载人员检测模型失败: {e}")
        return None


# 根据检测类型加载相应模型
if detection_type == "火灾烟雾检测":
    model_type = "best" if model_option.startswith("最佳模型") else "last"
    model = load_fire_smoke_model(model_type)
    fire_mode = True
else:
    model = load_person_detection_model()
    fire_mode = False

if model is None:
    st.stop()

# 设备选择逻辑
def get_device(option, gpu_available):
    """根据用户选择获取设备字符串"""
    if option == "CPU":
        return "cpu"
    elif option == "GPU":
        if gpu_available:
            return "cuda"
        else:
            st.warning("GPU不可用，将使用CPU进行推理")
            return "cpu"
    else:  # 自动选择
        if gpu_available:
            return None  # 让模型自动选择
        else:
            return "cpu"  # GPU不可用，强制使用CPU

device = get_device(device_option, gpu_available)

# ==================== 训练结果展示（仅火灾烟雾检测模式）====================
if fire_mode:
    if show_metrics:
        st.header("关键训练指标")
        metrics = load_training_metrics()
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("精确率 (Precision)", f"{metrics.get('precision', 0):.3f}")
            with col2:
                st.metric("召回率 (Recall)", f"{metrics.get('recall', 0):.3f}")
            with col3:
                st.metric("mAP@0.5", f"{metrics.get('mAP50', 0):.3f}")
            with col4:
                st.metric("mAP@0.5:0.95", f"{metrics.get('mAP50_95', 0):.3f}")
            st.caption(f"最终训练轮数: {metrics.get('epoch', 0)}")
        else:
            st.warning("无法加载训练指标")
        st.divider()

    # 显示训练结果图表
    if show_results:
        st.header("训练结果图表")

        # 图表文件路径
        charts_dir = "runs/detect/fire_smoke_optimized1"
        chart_files = {
            "训练曲线": "results.png",
            "混淆矩阵": "confusion_matrix.png",
            "归一化混淆矩阵": "confusion_matrix_normalized.png",
            "PR曲线": "BoxPR_curve.png",
            "P曲线": "BoxP_curve.png",
            "R曲线": "BoxR_curve.png",
            "F1曲线": "BoxF1_curve.png"
        }

        # 显示图表
        cols = st.columns(2)
        col_idx = 0

        for chart_name, chart_file in chart_files.items():
            chart_path = os.path.join(charts_dir, chart_file)
            if os.path.exists(chart_path):
                with cols[col_idx % 2]:
                    st.subheader(chart_name)
                    st.image(chart_path)
                col_idx += 1

    st.divider()


# ==================== 主检测区域 ====================
if fire_mode:
    st.header("🔍 火灾烟雾检测区域")
else:
    st.header("🚶 非法人员闯入检测区域")


def process_and_show_results(results, detection_type_fire, alarm_threshold_val):
    """处理检测结果并显示，返回是否触发报警"""
    annotated_img = results[0].plot()
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    # 显示检测结果
    st.image(annotated_img_rgb, caption="检测结果")

    # 显示检测统计信息
    detections = results[0].boxes
    alarm_triggered = False

    if detections is not None and len(detections) > 0:
        if detection_type_fire:
            # 火灾烟雾检测统计
            num_fire = sum(1 for cls in detections.cls if cls == 0)
            num_smoke = sum(1 for cls in detections.cls if cls == 1)

            st.success(f"检测完成！共检测到 {len(detections)} 个目标")
            st.info(f"🔥 火灾: {num_fire} 个 | 💨 烟雾: {num_smoke} 个")

            # 显示详细检测结果
            with st.expander("查看详细检测结果"):
                for i, box in enumerate(detections):
                    cls = int(box.cls)
                    conf = float(box.conf)
                    class_name = "Fire" if cls == 0 else "Smoke"
                    st.write(f"目标 {i+1}: {class_name} (置信度: {conf:.2f})")
        else:
            # 人员闯入检测统计
            st.success(f"检测完成！共检测到 {len(detections)} 个目标")

            # 统计各类别数量（人员检测用COCO，可能有其他类）
            person_count = 0
            other_counts = {}
            for cls_tensor in detections.cls:
                cls_id = int(cls_tensor)
                if cls_id == 0:
                    person_count += 1
                else:
                    class_name = results[0].names.get(cls_id, f"class_{cls_id}")
                    other_counts[class_name] = other_counts.get(class_name, 0) + 1

            st.info(f"🚶 人员: {person_count} 个")
            for name, count in other_counts.items():
                st.write(f"其他目标 ({name}): {count} 个")

            # 检查是否触发闯入报警
            if person_count >= alarm_threshold_val:
                alarm_triggered = True
                st.error(f"🚨 **非法闯入报警！** 检测到 **{person_count}** 个人员！")
                st.warning("⚠️ 检测到未授权人员进入，请注意安全！")

            # 显示详细检测结果
            with st.expander("查看详细检测结果"):
                for i, box in enumerate(detections):
                    cls = int(box.cls)
                    conf = float(box.conf)
                    class_name = results[0].names.get(cls, f"未知类别({cls})")
                    st.write(f"目标 {i+1}: {class_name} (置信度: {conf:.2f})")
    else:
        if detection_type_fire:
            st.warning("未检测到火灾或烟雾")
        else:
            st.info("✅ 未检测到人员闯入")

    return alarm_triggered


# ---------- 图片上传检测 ----------
if detection_mode == "图片上传":
    if fire_mode:
        st.subheader("图片上传检测（火灾烟雾）")
    else:
        st.subheader("图片上传检测（人员闯入）")

    uploaded_file = st.file_uploader(
        "上传图片 (支持 JPG, PNG, JPEG)",
        type=["jpg", "png", "jpeg"],
        help="上传图片进行检测"
    )

    if uploaded_file is not None:
        # 显示原图
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("原图")
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图片")

        with col2:
            st.subheader("检测结果")

            # 进行检测
            with st.spinner("正在检测..."):
                # 保存为临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    image.save(tmp_file, format='JPEG')
                    tmp_path = tmp_file.name

                try:
                    predict_args = {
                        'source': tmp_path,
                        'conf': conf_threshold,
                        'iou': 0.7,
                        'save': False,
                        'show': False
                    }
                    if device is not None:
                        predict_args['device'] = device
                    results = model.predict(**predict_args)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

                process_and_show_results(results, fire_mode, alarm_threshold if not fire_mode else 1)

# ---------- 摄像头实时检测 ----------
else:
    if fire_mode:
        st.subheader("摄像头实时检测（火灾烟雾）")
    else:
        st.subheader("摄像头实时检测（人员闯入）")

    st.info("注意：摄像头检测需要允许浏览器访问摄像头权限")

    # 使用streamlit的摄像头输入
    camera_img = st.camera_input("打开摄像头进行实时检测")

    if camera_img is not None:
        # 显示原图
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("摄像头画面")
            st.image(camera_img, caption="摄像头画面")

        with col2:
            st.subheader("检测结果")

            # 进行检测
            with st.spinner("正在检测..."):
                camera_image = Image.open(camera_img)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    camera_image.save(tmp_file, format='JPEG')
                    tmp_path = tmp_file.name

                try:
                    predict_args = {
                        'source': tmp_path,
                        'conf': conf_threshold,
                        'iou': 0.7,
                        'save': False,
                        'show': False
                    }
                    if device is not None:
                        predict_args['device'] = device
                    results = model.predict(**predict_args)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

                process_and_show_results(results, fire_mode, alarm_threshold if not fire_mode else 1)

# ==================== 页脚 ====================
st.divider()
st.markdown("""
<div style="text-align: center; color: gray;">
    <p>智能检测系统 &copy; 2024 | 基于 Ultralytics YOLOv8 | 支持火灾烟雾检测 & 非法人员闯入检测</p>
</div>
""", unsafe_allow_html=True)
