from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import onnxruntime as ort
import cv2
import yaml

app = Flask(__name__)

# 配置路径
MODEL_PATH = "best1.onnx"
CONFIG_PATH = "xinjiaonang.yaml"  # 替换为你的数据集配置文件
CONFIDENCE_THRESHOLD = 0.05
IOU_THRESHOLD = 0.4


# 加载类别名称
def load_class_names(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config["names"]


class_names = load_class_names(CONFIG_PATH)

# 初始化 ONNX Session
ort_session = ort.InferenceSession(MODEL_PATH)

# 获取输入输出名称
input_name = ort_session.get_inputs()[0].name
output_names = [o.name for o in ort_session.get_outputs()]

# 获取输入尺寸
input_shape = ort_session.get_inputs()[0].shape  # [1, 3, H, W]
_, _, INPUT_HEIGHT, INPUT_WIDTH = input_shape


def preprocess(image: Image.Image) -> (np.ndarray, int, int):
    """
    图像预处理：缩放、归一化、通道转换
    """
    img = image.convert("RGB")
    original_width, original_height = img.size
    img = img.resize((INPUT_WIDTH, INPUT_HEIGHT))  # 调整到模型输入大小
    img = np.array(img).astype(np.float32) / 255.0  # 归一化
    img = img.transpose(2, 0, 1)[np.newaxis, ...]  # NHWC -> NCHW
    return img.astype(np.float32), original_width, original_height


def postprocess(outputs: list, original_width: int, original_height: int, conf_threshold=0.25) -> list:
    """
    后处理函数，适用于输出形状为 [1, 39, 8400] 的 ONNX 模型，并包含 NMS：
    """
    output = outputs[0].squeeze()  # 去掉 batch 维度 => [39, 8400]

    boxes, scores, class_ids = [], [], []

    for i in range(output.shape[1]):
        box = output[:, i]
        x, y, w, h = box[:4]
        scores_per_box = box[4:]
        max_score_idx = np.argmax(scores_per_box)
        max_score = scores_per_box[max_score_idx]

        if max_score >= conf_threshold:
            cls_id = int(max_score_idx)

            # 还原坐标到原始图像尺寸
            x1 = (x - w / 2) * original_width / INPUT_WIDTH
            y1 = (y - h / 2) * original_height / INPUT_HEIGHT
            x2 = (x + w / 2) * original_width / INPUT_WIDTH
            y2 = (y + h / 2) * original_height / INPUT_HEIGHT

            boxes.append([x1, y1, x2 - x1, y2 - y1])  # xywh
            scores.append(max_score)
            class_ids.append(cls_id)

    # 应用 NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, IOU_THRESHOLD)

    results = []
    if len(indices) > 0:
        for idx in indices.flatten():
            cls_id = class_ids[idx]
            name = class_names[cls_id] if cls_id < len(class_names) else "unknown"

            # 将坐标转换为xyxy格式
            box = boxes[idx]
            x1, y1, width, height = box
            x2, y2 = x1 + width, y1 + height

            results.append({
                "coor": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],  # xyxy
                "conf": round(float(scores[idx]), 3),
                "class": name
            })

    return results


@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream)
        input_data, original_width, original_height = preprocess(image)
        outputs = ort_session.run(output_names, {input_name: input_data})

        detections = postprocess(outputs, original_width, original_height, CONFIDENCE_THRESHOLD)
        return jsonify(detections)
    except Exception as e:
        import traceback
        print("Error occurred:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)