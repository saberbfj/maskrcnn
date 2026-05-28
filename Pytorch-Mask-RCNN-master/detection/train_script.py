import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from .engine import train_one_epoch, evaluate
from . import utils
from . import transforms as T


class ChannelAttention(torch.nn.Module):
    """通道注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc1 = torch.nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(torch.nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = torch.nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x) * x


def get_unique_filename(directory, base_name, extension):
    """生成唯一的文件名，如果文件已存在则添加角标"""
    counter = 1
    while True:
        if counter == 1:
            filename = f"{base_name}{extension}"
        else:
            filename = f"{base_name}{counter}{extension}"
        full_path = os.path.join(directory, filename)
        if not os.path.exists(full_path):
            return filename
        counter += 1


class PennFudanDataset(object):
    def __init__(self, root, split, transforms):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, split, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, split, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.split, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, self.split, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        valid_indices = []
        
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                valid_indices.append(i)
            else:
                print(f"警告: 跳过无效边界框 [{xmin}, {ymin}, {xmax}, {ymax}] 在图像 {self.imgs[idx]}")

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)
            masks = torch.as_tensor(masks[valid_indices], dtype=torch.uint8)

        image_id = torch.tensor([idx])
        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def train_model(dataset_dir, save_dir, user_id, file_manager, update_callback=None, stop_check=None):
    """训练模型并返回模型路径"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if update_callback:
        update_callback(0, f"使用设备: {device}")
    
    num_classes = 2
    dataset = PennFudanDataset(dataset_dir, 'train', get_transform(train=True))
    dataset_test = PennFudanDataset(dataset_dir, 'val', get_transform(train=False))
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    num_epochs = 50
    best_map = 0.0
    patience = 10
    no_improve_count = 0
    best_model_state = None
    was_stopped = False
    
    for epoch in range(num_epochs):
        if stop_check and stop_check():
            was_stopped = True
            if update_callback:
                update_callback(100, "训练已停止")
            break
        
        if update_callback:
            update_callback(int((epoch / 50) * 100), f"开始第 {epoch+1} 轮训练")
        
        metric_logger, loss_data = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, stop_check=stop_check)
        
        # 检查训练过程中是否被停止
        if stop_check and stop_check():
            was_stopped = True
            if update_callback:
                update_callback(100, "训练已停止")
            break
        
        if update_callback:
            update_callback(int(((epoch + 0.5) / 50) * 100), f"Epoch {epoch+1} 训练完成", loss_data)
        
        if stop_check and stop_check():
            was_stopped = True
            if update_callback:
                update_callback(100, "训练已停止")
            break
        
        lr_scheduler.step()
        
        if update_callback:
            update_callback(int(((epoch + 0.5) / 50) * 100), "开始评估...")
        
        coco_evaluator = evaluate(model, data_loader_test, device=device, stop_check=stop_check)
        
        # 检查评估过程中是否被停止
        if stop_check and stop_check():
            was_stopped = True
            if update_callback:
                update_callback(100, "训练已停止")
            break
        
        current_map = coco_evaluator.coco_eval['segm'].stats[0]
        
        if update_callback:
            update_callback(int(((epoch + 1) / 50) * 100), f"第 {epoch+1} 轮评估完成，mAP: {current_map:.4f}")
        
        if current_map > best_map + 1e-4:  # 添加1e-4的阈值，解决浮点精度问题
            best_map = current_map
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
            if update_callback:
                update_callback(int(((epoch + 1) / 50) * 100), f"获得最佳模型，mAP: {best_map:.4f}")
        else:
            no_improve_count += 1
            if update_callback:
                update_callback(int(((epoch + 1) / 50) * 100), f"mAP未提升，已连续 {no_improve_count}/{patience} 轮")
            
            if no_improve_count >= patience:
                if update_callback:
                    update_callback(100, f"早停触发，连续 {patience} 轮mAP未提升，停止训练")
                break
        
        if stop_check and stop_check():
            was_stopped = True
            if update_callback:
                update_callback(100, "训练已停止")
            break
    
    # 如果被停止，不保存任何模型
    if was_stopped:
        if update_callback:
            update_callback(100, "训练已停止，不保存模型权重")
        return None
    
    # 确定保存目录
    if not save_dir:
        # 如果没有指定保存目录，使用默认目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(project_root, "model")
    else:
        model_dir = save_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 从数据集目录获取类别名称
    class_name = os.path.basename(dataset_dir)
    
    # 生成文件名
    base_name = f"MaskRCNN_best_{class_name}"
    final_base_name = f"MaskRCNN_{class_name}"
    
    best_model_filename = get_unique_filename(model_dir, base_name, ".pth")
    final_model_filename = get_unique_filename(model_dir, final_base_name, ".pth")
    
    best_model_path = os.path.join(model_dir, best_model_filename)
    final_model_path = os.path.join(model_dir, final_model_filename)
    
    # 保存最佳模型（使用训练过程中保存的最佳权重）
    if best_model_state is not None:
        best_save_data = {
            'model_state_dict': best_model_state,
            'num_classes': num_classes,
            'class_labels': [1],
            'best_map': best_map
        }
    else:
        best_save_data = {
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
            'class_labels': [1]
        }
    torch.save(best_save_data, best_model_path)
    
    # 保存最终模型
    final_save_data = {
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_labels': [1]
    }
    torch.save(final_save_data, final_model_path)
    
    # 记录模型
    file_manager.add_model(
        user_id=user_id,
        filename=best_model_filename,
        class_name=class_name,
        model_type="best",
        file_path=best_model_path
    )
    
    file_manager.add_model(
        user_id=user_id,
        filename=final_model_filename,
        class_name=class_name,
        model_type="final",
        file_path=final_model_path
    )
    
    if update_callback:
        update_callback(100, f"模型训练完成，最佳模型已保存到: {best_model_path}")
    
    return final_model_path, best_model_path


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2
    dataset_root = 'PennFudanPed'
    dataset = PennFudanDataset(dataset_root, 'train', get_transform(train=True))
    dataset_test = PennFudanDataset(dataset_root, 'val', get_transform(train=False))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 50
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
    
    print("保存模型...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, "model")
    os.makedirs(model_dir, exist_ok=True)
    unique_filename = get_unique_filename(model_dir, "MaskRCNN", ".pth")
    model_path = os.path.join(model_dir, unique_filename)
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")


if __name__ == "__main__":
    main()
