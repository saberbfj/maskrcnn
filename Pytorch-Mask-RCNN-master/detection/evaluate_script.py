import os
import torch
import numpy as np
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from .engine import evaluate
from . import utils
from . import transforms as T


class TestDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

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


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def save_evaluation_results(coco_evaluator, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"evaluation_results_{timestamp}.txt")
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('Class      Images   Labels          mAP50   mAP50-95\n')
        f.write('all        {}        {}          {:.4f}      {:.4f}\n'.format(
            coco_evaluator.coco_eval['bbox'].params.imgIds.__len__(),
            coco_evaluator.coco_eval['bbox'].params.maxDets[2],
            coco_evaluator.coco_eval['bbox'].stats[1],
            coco_evaluator.coco_eval['bbox'].stats[0]
        ))
        
        f.write('\nMask       Images   Labels          mAP50   mAP50-95\n')
        f.write('all        {}        {}          {:.4f}      {:.4f}\n'.format(
            coco_evaluator.coco_eval['segm'].params.imgIds.__len__(),
            coco_evaluator.coco_eval['segm'].params.maxDets[2],
            coco_evaluator.coco_eval['segm'].stats[1],
            coco_evaluator.coco_eval['segm'].stats[0]
        ))
        
        f.write('\nDetailed Metrics:\n')
        f.write('Bounding Box:\n')
        bbox_stats = coco_evaluator.coco_eval['bbox'].stats
        f.write('  mAP50: {:.4f}\n'.format(bbox_stats[1]))
        f.write('  mAP50-95: {:.4f}\n'.format(bbox_stats[0]))
        f.write('  mAP75: {:.4f}\n'.format(bbox_stats[2]))
        f.write('  AP_small: {:.4f}\n'.format(bbox_stats[3]))
        f.write('  AP_medium: {:.4f}\n'.format(bbox_stats[4]))
        f.write('  AP_large: {:.4f}\n'.format(bbox_stats[5]))
        f.write('  AR_max1: {:.4f}\n'.format(bbox_stats[6]))
        f.write('  AR_max10: {:.4f}\n'.format(bbox_stats[7]))
        f.write('  AR_max100: {:.4f}\n'.format(bbox_stats[8]))
        
        f.write('\nInstance Segmentation:\n')
        segm_stats = coco_evaluator.coco_eval['segm'].stats
        f.write('  mAP50: {:.4f}\n'.format(segm_stats[1]))
        f.write('  mAP50-95: {:.4f}\n'.format(segm_stats[0]))
        f.write('  mAP75: {:.4f}\n'.format(segm_stats[2]))
        f.write('  AP_small: {:.4f}\n'.format(segm_stats[3]))
        f.write('  AP_medium: {:.4f}\n'.format(segm_stats[4]))
        f.write('  AP_large: {:.4f}\n'.format(segm_stats[5]))
        f.write('  AR_max1: {:.4f}\n'.format(segm_stats[6]))
        f.write('  AR_max10: {:.4f}\n'.format(segm_stats[7]))
        f.write('  AR_max100: {:.4f}\n'.format(segm_stats[8]))
    
    return save_path


def evaluate_test(test_dir, weights_path, save_dir):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        num_classes = checkpoint.get('num_classes', 2)
        print(f"加载模型包含 {num_classes} 个类别")
    else:
        model_state_dict = checkpoint
        if 'roi_heads.box_predictor.cls_score.weight' in model_state_dict:
            num_classes = model_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]
            print(f"从模型权重推断出 {num_classes} 个类别")
        else:
            num_classes = 2
            print("使用默认类别设置（2个类别）")
    
    dataset = TestDataset(test_dir, get_transform(train=False))
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    
    model = get_model_instance_segmentation(num_classes)
    model.load_state_dict(model_state_dict)
    model.to(device)
    
    coco_evaluator = evaluate(model, data_loader, device=device)
    
    save_path = save_evaluation_results(coco_evaluator, save_dir)
    
    return coco_evaluator, save_path


if __name__ == "__main__":
    import argparse
    
    parse = argparse.ArgumentParser(description="评估测试集")
    parse.add_argument('--test-dir', type=str, required=True, help='测试集目录路径')
    parse.add_argument('--weights-path', type=str, required=True, help='模型权重文件路径')
    parse.add_argument('--save-dir', type=str, default='evaluation_results', help='评估结果保存目录')
    args = parse.parse_args()
    
    print(f"开始评估测试集: {args.test_dir}")
    print(f"使用模型权重: {args.weights_path}")
    print(f"保存评估结果到: {args.save_dir}")
    
    coco_evaluator, save_path = evaluate_test(args.test_dir, args.weights_path, args.save_dir)
    
    print("\n评估结果:")
    print(f"bbox mAP @ IoU=0.50:0.95: {coco_evaluator.coco_eval['bbox'].stats[0]:.4f}")
    print(f"bbox mAP @ IoU=0.50: {coco_evaluator.coco_eval['bbox'].stats[1]:.4f}")
    print(f"bbox mAP @ IoU=0.75: {coco_evaluator.coco_eval['bbox'].stats[2]:.4f}")
    print(f"segm mAP @ IoU=0.50:0.95: {coco_evaluator.coco_eval['segm'].stats[0]:.4f}")
    print(f"segm mAP @ IoU=0.50: {coco_evaluator.coco_eval['segm'].stats[1]:.4f}")
    print(f"segm mAP @ IoU=0.75: {coco_evaluator.coco_eval['segm'].stats[2]:.4f}")
    print(f"\n完整评估结果已保存到: {save_path}")
