import os
import torch
import numpy as np
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from .draw_box_utils import draw_objs, draw_objs_with_colors


def get_model_instance_segmentation(num_classes, use_pretrained=False):
    if use_pretrained:
        try:
            weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        except:
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    else:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def predict_image(img_path, model_paths, box_conf=0.3, mask_conf=0.3):
    """使用多个模型预测图片并返回结果"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    try:
        ori_img = Image.open(img_path).convert('RGB')
        data_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img = data_transform(ori_img)
    except Exception as e:
        # 如果图片加载失败，返回原始图片和空结果
        import traceback
        print(f"图片加载失败: {str(e)}")
        print(traceback.format_exc())
        return Image.new('RGB', (512, 512), color='white'), []

    all_boxes = []
    all_classes = []
    all_scores = []
    all_masks = []
    all_indices = {}
    
    model_results = []

    for i, model_path in enumerate(model_paths):
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                class_labels = checkpoint.get('class_labels', [1])
                num_classes = checkpoint.get('num_classes', 2)
            else:
                model_state_dict = checkpoint
                class_labels = [1]
                if 'roi_heads.box_predictor.cls_score.weight' in model_state_dict:
                    num_classes = model_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]
                else:
                    num_classes = 2
            
            model = get_model_instance_segmentation(num_classes)
            model.load_state_dict(model_state_dict)
            model.to(device)
            model.eval()

            # 预热模型
            for _ in range(2):
                try:
                    _ = model([img.to(device)])
                except Exception:
                    pass

            predictions = model([img.to(device)])[0]

            boxes = predictions["boxes"].to("cpu").detach().numpy()
            classes = predictions["labels"].to("cpu").detach().numpy()
            scores = predictions["scores"].to("cpu").detach().numpy()
            masks = predictions["masks"].to("cpu").detach().numpy()
            masks = np.squeeze(masks, axis=1)

            mask = scores >= box_conf
            boxes = boxes[mask]
            classes = classes[mask]
            scores = scores[mask]
            masks = masks[mask]

            num_instances = len(boxes)
            
            # 统计每种实例的数量
            instance_counts = {}
            # 从模型文件名中提取类别名称
            model_basename = os.path.basename(model_path)
            # 例如: MaskRCNN_best_orange.pth -> orange
            if 'best_' in model_basename:
                class_name = model_basename.split('best_')[1].split('.')[0]
            elif '_' in model_basename:
                # 尝试其他格式
                parts = model_basename.split('_')
                class_name = parts[-1].split('.')[0] if len(parts) > 1 else 'object'
            else:
                class_name = 'object'
            
            # 只统计属于当前模型类别的实例
            # 注意：这里假设模型只应该检测它自己的类别
            # 例如，apple模型只应该检测apple，orange模型只应该检测orange
            # 所以我们直接将实例数量设置为1（如果有检测到的话）
            # 这样可以避免模型检测到其他类别的情况
            instance_counts[class_name] = 1 if len(boxes) > 0 else 0
            num_instances = 1 if len(boxes) > 0 else 0
            
            model_results.append({
                'model_path': model_path,
                'num_instances': num_instances,
                'instance_counts': instance_counts
            })

            if len(boxes) > 0:
                model_prefix = f"model{i+1}_"
                # 使用从文件名提取的类别名称
                for label_id in class_labels:
                    all_indices[model_prefix + str(label_id)] = f"{class_name}"
                
                prefixed_classes = [f"{model_prefix}{class_name}" for _ in classes]
                
                all_boxes.extend(boxes)
                all_classes.extend(prefixed_classes)
                all_scores.extend(scores)
                all_masks.extend(masks)
        except Exception as e:
            # 单个模型失败不影响其他模型
            import traceback
            print(f"模型 {model_path} 预测失败: {str(e)}")
            print(traceback.format_exc())
            # 添加失败模型的结果
            model_results.append({
                'model_path': model_path,
                'num_instances': 0,
                'instance_counts': {}
            })

    if len(all_boxes) == 0:
        return ori_img, model_results

    try:
        # 应用非极大值抑制（NMS）来避免重复预测
        # 首先将所有预测结果按置信度排序
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        sorted_indices = np.argsort(all_scores)[::-1]
        all_boxes = all_boxes[sorted_indices]
        all_classes = [all_classes[i] for i in sorted_indices]
        all_scores = all_scores[sorted_indices]
        if len(all_masks) > 0:
            all_masks = np.array(all_masks)
            all_masks = all_masks[sorted_indices]
        
        # 执行NMS
        keep_boxes = []
        keep_classes = []
        keep_scores = []
        keep_masks = []
        iou_threshold = 0.5
        
        # 复制原始数据
        boxes_copy = all_boxes.copy()
        classes_copy = all_classes.copy()
        scores_copy = all_scores.copy()
        masks_copy = all_masks.copy() if len(all_masks) > 0 else []
        
        while len(boxes_copy) > 0:
            # 找到置信度最高的预测
            max_score_idx = np.argmax(scores_copy)
            current_box = boxes_copy[max_score_idx]
            current_class = classes_copy[max_score_idx]
            current_score = scores_copy[max_score_idx]
            current_mask = masks_copy[max_score_idx] if len(masks_copy) > 0 else None
            
            # 保留这个预测
            keep_boxes.append(current_box)
            keep_classes.append(current_class)
            keep_scores.append(current_score)
            if current_mask is not None:
                keep_masks.append(current_mask)
            
            # 计算与其他预测的IoU
            rest_boxes = np.delete(boxes_copy, max_score_idx, axis=0)
            rest_classes = [classes_copy[i] for i in range(len(classes_copy)) if i != max_score_idx]
            rest_scores = np.delete(scores_copy, max_score_idx)
            rest_masks = [masks_copy[i] for i in range(len(masks_copy)) if i != max_score_idx] if len(masks_copy) > 0 else []
            
            if len(rest_boxes) == 0:
                break
            
            # 计算IoU
            ious = []
            for box in rest_boxes:
                # 计算交集
                x1 = max(current_box[0], box[0])
                y1 = max(current_box[1], box[1])
                x2 = min(current_box[2], box[2])
                y2 = min(current_box[3], box[3])
                
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                area2 = (box[2] - box[0]) * (box[3] - box[1])
                union = area1 + area2 - intersection
                
                iou = intersection / union if union > 0 else 0
                ious.append(iou)
            
            # 保留IoU小于阈值的预测
            ious = np.array(ious)
            keep_mask = ious < iou_threshold
            
            boxes_copy = rest_boxes[keep_mask]
            classes_copy = [rest_classes[i] for i, keep in enumerate(keep_mask) if keep]
            scores_copy = rest_scores[keep_mask]
            if len(rest_masks) > 0:
                keep_masks_temp = [rest_masks[i] for i, keep in enumerate(keep_mask) if keep]
                masks_copy = keep_masks_temp
        
        # 更新为NMS处理后的结果
        all_boxes = np.array(keep_boxes) if keep_boxes else np.array([])
        all_classes = keep_classes
        all_scores = np.array(keep_scores) if keep_scores else np.array([])
        all_masks = np.array(keep_masks) if keep_masks else []
        
        # 为不同模型使用不同颜色
        # 使用model_index * 100作为颜色索引的偏移量，确保不同模型即使类别相同也使用不同颜色
        class_to_id = {}
        id_to_class = {}
        current_id = 1
        numeric_classes = []
        
        for cls in all_classes:
            if cls not in class_to_id:
                # 计算颜色索引：使用类别名称中的模型索引来分配不同的基础颜色
                class_to_id[cls] = current_id
                id_to_class[current_id] = cls
                current_id += 1
            numeric_classes.append(class_to_id[cls])
        
        # 为不同模型使用不同的颜色集合
        # STANDARD_COLORS有大量颜色，我们可以通过模型索引来选择不同的颜色子集
        from .draw_box_utils import STANDARD_COLORS
        
        # 创建category_index，添加颜色信息
        category_index = {}
        for cls_name, cls_id in class_to_id.items():
            # 从类别名称中提取模型索引（如果有）
            if "model" in cls_name:
                model_idx = int(cls_name.split("model")[1].split("_")[0]) - 1
                # 使用模型索引来选择不同范围的颜色
                color_idx = (model_idx * 10) % len(STANDARD_COLORS)
                category_index[str(cls_id)] = {"name": cls_name, "color": STANDARD_COLORS[color_idx]}
            else:
                category_index[str(cls_id)] = {"name": cls_name, "color": STANDARD_COLORS[cls_id % len(STANDARD_COLORS)]}

        # 确保变量类型正确
        if not isinstance(all_boxes, np.ndarray):
            all_boxes = np.array(all_boxes)
        numeric_classes = np.array(numeric_classes)
        if not isinstance(all_scores, np.ndarray):
            all_scores = np.array(all_scores)
        if len(all_masks) > 0 and not isinstance(all_masks, np.ndarray):
            all_masks = np.array(all_masks)

        # 确保所有参数都有正确的值
        if len(all_boxes) > 0 and len(numeric_classes) > 0 and len(all_scores) > 0:
            print(f"绘制检测框: {len(all_boxes)} 个实例")
            print(f"类别: {numeric_classes}")
            print(f"分数: {all_scores}")
            print(f"类别索引: {category_index}")
            
            plot_img = draw_objs_with_colors(ori_img,
                                 boxes=all_boxes,
                                 classes=numeric_classes,
                                 scores=all_scores,
                                 masks=all_masks if len(all_masks) > 0 else None,
                                 category_index=category_index,
                                 box_thresh=0.1,  # 使用较低的阈值确保显示
                                 mask_thresh=mask_conf,
                                 line_thickness=5,  # 增加线条宽度
                                 font='arial.ttf',
                                 font_size=16)
        else:
            print("没有检测到实例")
            plot_img = ori_img
    except Exception as e:
        # 如果绘制失败，返回原始图片
        import traceback
        print(f"绘制结果失败: {str(e)}")
        print(traceback.format_exc())
        plot_img = ori_img

    return plot_img, model_results