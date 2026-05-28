from fastapi import APIRouter, Form, UploadFile, File
from fastapi.responses import JSONResponse
import os
import base64
import json
import tempfile
import shutil
from io import BytesIO
from PIL import Image
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from datetime import datetime

router = APIRouter()

def get_model_instance_segmentation(num_classes):
    """获取实例分割模型"""
    try:
        weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    except:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

def load_model(weights_path, device):
    """加载模型"""
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        num_classes = checkpoint.get('num_classes', 2)
    else:
        model_state_dict = checkpoint
        if 'roi_heads.box_predictor.cls_score.weight' in model_state_dict:
            num_classes = model_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]
        else:
            num_classes = 2
    
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    return model

def predict_image(image, model, box_conf=0.3, mask_conf=0.3):
    """预测单张图片"""
    import numpy as np
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    img = image.convert("RGB")
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        prediction = model([img_tensor])
    
    boxes = prediction[0]['boxes'].cpu().numpy()
    masks = prediction[0]['masks'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    
    result_boxes = boxes[scores >= box_conf]
    result_masks = masks[scores >= box_conf]
    result_labels = labels[scores >= box_conf]
    result_scores = scores[scores >= box_conf]
    
    return {
        'boxes': result_boxes.tolist(),
        'masks': result_masks.tolist(),
        'labels': result_labels.tolist(),
        'scores': result_scores.tolist()
    }

def get_model_class_names(model_path):
    """根据模型路径获取类别名称"""
    model_name = os.path.basename(model_path).lower()
    if 'apple' in model_name:
        return ['background', 'apple']
    elif 'orange' in model_name:
        return ['background', 'orange']
    elif 'banana' in model_name:
        return ['background', 'banana']
    else:
        # 默认使用COCO类别
        return ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']


def draw_predictions(image, predictions, class_names=None):
    """绘制预测结果"""
    import numpy as np
    from PIL import ImageDraw, ImageFilter
    
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    
    boxes = predictions.get('boxes', [])
    masks = predictions.get('masks', [])
    labels = predictions.get('labels', [])
    scores = predictions.get('scores', [])
    
    if class_names is None:
        # 默认使用COCO类别
        class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                       'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                       'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
                       'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                       'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                       'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                       'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                       'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                       'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']
    
    instance_counts = {}
    
    # 颜色映射 - 按类别分配颜色
    class_colors = {
        'apple': 'red',
        'orange': 'orange',
        'banana': 'yellow',
        'person': 'blue',
        'bicycle': 'green',
        'car': 'purple',
        'motorcycle': 'cyan',
        'airplane': 'magenta'
    }
    
    # 默认颜色列表，用于未在class_colors中定义的类别
    default_colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta', 'orange']
    
    for i, (box, mask, label, score) in enumerate(zip(boxes, masks, labels, scores)):
        x1, y1, x2, y2 = box
        label_name = class_names[label] if label < len(class_names) else f"class_{label}"
        
        if label_name not in instance_counts:
            instance_counts[label_name] = 0
        instance_counts[label_name] += 1
        
        # 选择颜色 - 按类别分配
        if label_name in class_colors:
            color = class_colors[label_name]
        else:
            # 对于未定义的类别，使用基于标签ID的默认颜色
            color = default_colors[label % len(default_colors)]
        
        # 绘制掩码
        if mask is not None and len(mask) > 0:
            try:
                # 处理掩码数据 - 处理列表格式的掩码
                import numpy as np
                
                # 检查mask的结构
                if isinstance(mask, list):
                    # 尝试将列表转换回numpy数组
                    mask_array = np.array(mask)
                    # 确保mask_array的形状是(1, H, W)
                    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
                        mask_array = mask_array[0]
                    elif mask_array.ndim == 2:
                        pass  # 已经是(H, W)形状
                    else:
                        continue  # 形状不符合预期，跳过
                else:
                    # 假设是numpy数组
                    mask_array = mask[0] if mask.ndim == 3 else mask
                
                # 阈值处理
                mask_array = (mask_array > 0.5).astype(np.uint8)
                
                # 创建掩码图像
                mask_img = Image.fromarray(mask_array * 255, mode='L')
                
                # 调整掩码大小以匹配原图
                if mask_img.size != img.size:
                    mask_img = mask_img.resize(img.size)
                
                # 创建彩色掩码 - 使用更高效的方法
                color_mask = Image.new('RGBA', img.size, (0, 0, 0, 0))
                
                # 快速填充掩码区域
                mask_np = np.array(mask_img)
                
                # 根据颜色设置RGB值
                if color == 'red':
                    rgb = (255, 0, 0)
                elif color == 'green':
                    rgb = (0, 255, 0)
                elif color == 'blue':
                    rgb = (0, 0, 255)
                elif color == 'yellow':
                    rgb = (255, 255, 0)
                elif color == 'purple':
                    rgb = (128, 0, 128)
                elif color == 'cyan':
                    rgb = (0, 255, 255)
                elif color == 'magenta':
                    rgb = (255, 0, 255)
                elif color == 'orange':
                    rgb = (255, 165, 0)
                else:
                    rgb = (255, 0, 0)  # 默认红色
                
                # 创建RGB+Alpha数组
                color_array = np.zeros((img.height, img.width, 4), dtype=np.uint8)
                color_array[mask_np > 128] = (*rgb, 64)  # 设置半透明
                
                # 转换为图像
                color_mask = Image.fromarray(color_array, mode='RGBA')
                
                # 将掩码叠加到原图
                img = Image.alpha_composite(img.convert('RGBA'), color_mask)
                img = img.convert('RGB')
            except Exception as e:
                # 打印错误信息以便调试
                print(f"掩码绘制失败: {str(e)}")
                pass  # 如果掩码绘制失败，跳过
        
        # 重新创建绘图对象，确保在掩码之上绘制
        draw = ImageDraw.Draw(img)
        
        # 绘制边界框 - 使用更明显的颜色和宽度
        # 为边界框选择与掩码对比明显的颜色
        if color == 'red':
            box_color = 'white'  # 红色掩码用白色边界框
        elif color == 'orange':
            box_color = 'blue'  # 橙色掩码用蓝色边界框
        elif color == 'yellow':
            box_color = 'blue'  # 黄色掩码用蓝色边界框
        elif color == 'blue':
            box_color = 'white'  # 蓝色掩码用白色边界框
        elif color == 'green':
            box_color = 'white'  # 绿色掩码用白色边界框
        elif color == 'purple':
            box_color = 'yellow'  # 紫色掩码用黄色边界框
        elif color == 'cyan':
            box_color = 'red'  # 青色掩码用红色边界框
        elif color == 'magenta':
            box_color = 'green'  # 洋红色掩码用绿色边界框
        else:
            box_color = 'white'  # 默认用白色边界框
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)  # 增加线宽到3
        text = f"{label_name}: {score:.2f}"
        # 绘制文本标签 - 使用与边界框相同的颜色
        draw.text((x1, y1 - 15), text, fill=box_color)  # 调整文本位置
    
    return img, instance_counts

@router.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    model_paths: str = Form(...),
    save_dir: str = Form(None),
    box_conf: str = Form("0.3"),
    mask_conf: str = Form("0.3"),
    username: str = Form(...),
    password: str = Form(...)
):
    """预测图片"""
    from server.main import user_manager, file_manager
    
    success, message = user_manager.login_user(username, password)
    if not success:
        return JSONResponse(status_code=401, content={"error": message})
    
    user_id = user_manager.get_user_id(username)
    model_paths_list = json.loads(model_paths)
    box_conf = float(box_conf)
    mask_conf = float(mask_conf)
    
    temp_dir = None
    image_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_file.flush()
            image_path = temp_file.name
        
        img = Image.open(image_path).convert("RGB")
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        model_results = []
        all_instance_counts = {}
        
        # 合并所有模型的预测结果
        all_boxes = []
        all_masks = []
        all_labels = []
        all_scores = []
        
        # 为每个模型获取对应的类别名称
        model_class_mappings = {}
        for model_path in model_paths_list:
            if not os.path.exists(model_path):
                continue
            model_class_mappings[model_path] = get_model_class_names(model_path)
        
        # 确定合并后的类别名称列表
        # 对于多个模型的情况，使用默认的COCO类别
        merged_class_names = get_model_class_names(model_paths_list[0]) if model_paths_list else None
        
        for model_path in model_paths_list:
            if not os.path.exists(model_path):
                continue
            
            model = load_model(model_path, device)
            result = predict_image(img, model, box_conf, mask_conf)
            
            model_name = os.path.basename(model_path)
            model_results.append({
                'model': model_name,
                'boxes': len(result['boxes']),
                'masks': len(result['masks']),
                'labels': result['labels'],
                'scores': result['scores']
            })
            
            # 收集所有模型的结果
            all_boxes.extend(result['boxes'])
            all_masks.extend(result['masks'])
            all_labels.extend(result['labels'])
            all_scores.extend(result['scores'])
            
            # 使用模型特定的类别名称
            class_names = model_class_mappings[model_path]
            _, instance_counts = draw_predictions(img, result, class_names)
            
            for class_name, count in instance_counts.items():
                if class_name not in all_instance_counts:
                    all_instance_counts[class_name] = {}
                all_instance_counts[class_name][model_name] = count
        
        # 合并后的预测结果
        merged_result = {
            'boxes': all_boxes,
            'masks': all_masks,
            'labels': all_labels,
            'scores': all_scores
        }
        
        result_img, _ = draw_predictions(img, merged_result, merged_class_names)
        
        buffered = BytesIO()
        result_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            result_path = os.path.join(save_dir, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            result_img.save(result_path)
            
            file_manager.add_prediction(
                user_id=user_id,
                filename=image.filename,
                model_path=",".join(model_paths_list),
                box_conf=box_conf,
                mask_conf=mask_conf,
                num_instances=len(result['boxes']),
                instance_counts=all_instance_counts,
                result_path=result_path
            )
        
        return {
            "success": True,
            "image": img_str,
            "model_results": model_results
        }
        
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

@router.post("/api/predict/camera")
def predict_camera(
    image_base64: str = Form(...),
    model_paths: str = Form(...),
    box_conf: str = Form("0.3"),
    mask_conf: str = Form("0.3"),
    username: str = Form(...),
    password: str = Form(...)
):
    """预测摄像头图像"""
    from server.main import user_manager
    
    success, message = user_manager.login_user(username, password)
    if not success:
        return JSONResponse(status_code=401, content={"error": message})
    
    model_paths_list = json.loads(model_paths)
    box_conf = float(box_conf)
    mask_conf = float(mask_conf)
    
    image_data = base64.b64decode(image_base64)
    img = Image.open(BytesIO(image_data)).convert("RGB")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model_results = []
    all_instance_counts = {}
    
    # 合并所有模型的预测结果
    all_boxes = []
    all_masks = []
    all_labels = []
    all_scores = []
    
    # 为每个模型获取对应的类别名称
    model_class_mappings = {}
    for model_path in model_paths_list:
        if not os.path.exists(model_path):
            continue
        model_class_mappings[model_path] = get_model_class_names(model_path)
    
    # 确定合并后的类别名称列表
    # 对于多个模型的情况，使用默认的COCO类别
    merged_class_names = get_model_class_names(model_paths_list[0]) if model_paths_list else None
    
    for model_path in model_paths_list:
        if not os.path.exists(model_path):
            continue
        
        model = load_model(model_path, device)
        result = predict_image(img, model, box_conf, mask_conf)
        
        model_name = os.path.basename(model_path)
        model_results.append({
            'model': model_name,
            'boxes': len(result['boxes']),
            'masks': len(result['masks']),
            'labels': result['labels'],
            'scores': result['scores']
        })
        
        # 收集所有模型的结果
        all_boxes.extend(result['boxes'])
        all_masks.extend(result['masks'])
        all_labels.extend(result['labels'])
        all_scores.extend(result['scores'])
        
        # 使用模型特定的类别名称
        class_names = model_class_mappings[model_path]
        _, instance_counts = draw_predictions(img, result, class_names)
        
        for class_name, count in instance_counts.items():
            if class_name not in all_instance_counts:
                all_instance_counts[class_name] = {}
            all_instance_counts[class_name][model_name] = count
    
    # 合并后的预测结果
    merged_result = {
        'boxes': all_boxes,
        'masks': all_masks,
        'labels': all_labels,
        'scores': all_scores
    }
    
    result_img, _ = draw_predictions(img, merged_result, merged_class_names)
    
    buffered = BytesIO()
    result_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "success": True,
        "image": img_str,
        "model_results": model_results
    }
