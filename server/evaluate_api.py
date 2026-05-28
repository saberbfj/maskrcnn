from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
import os

router = APIRouter()

def evaluate_test(test_dir, weights_path, save_dir):
    """执行模型评估"""
    from detection.evaluate_script import evaluate_test as do_evaluate
    coco_evaluator, save_path = do_evaluate(
        test_dir=test_dir,
        weights_path=weights_path,
        save_dir=save_dir
    )
    return coco_evaluator, save_path

@router.post("/api/evaluate")
def evaluate(
    test_dir: str = Form(...),
    weights_path: str = Form(...),
    save_dir: str = Form(None),
    username: str = Form(...),
    password: str = Form(...)
):
    """评估模型"""
    from server.main import user_manager, file_manager
    
    success, message = user_manager.login_user(username, password)
    if not success:
        return JSONResponse(status_code=401, content={"error": message})
    
    user_id = user_manager.get_user_id(username)
    
    if not save_dir:
        return JSONResponse(status_code=400, content={"error": "请指定保存目录"})
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    coco_evaluator, save_path = evaluate_test(
        test_dir=test_dir,
        weights_path=weights_path,
        save_dir=save_dir
    )
    
    if coco_evaluator:
        bbox_stats = coco_evaluator.coco_eval['bbox'].stats
        segm_stats = coco_evaluator.coco_eval['segm'].stats
        file_manager.add_evaluation(
            user_id=user_id,
            model_path=weights_path,
            bbox_map50=bbox_stats[1],
            bbox_map50_95=bbox_stats[0],
            bbox_map75=bbox_stats[2],
            segm_map50=segm_stats[1],
            segm_map50_95=segm_stats[0],
            segm_map75=segm_stats[2]
        )
    
    bbox_stats = coco_evaluator.coco_eval['bbox'].stats if coco_evaluator else []
    segm_stats = coco_evaluator.coco_eval['segm'].stats if coco_evaluator else []
    
    evaluation_data = {
        "bbox_map50_95": bbox_stats[0] if len(bbox_stats) > 0 else 0,
        "bbox_map50": bbox_stats[1] if len(bbox_stats) > 1 else 0,
        "bbox_map75": bbox_stats[2] if len(bbox_stats) > 2 else 0,
        "bbox_ap_small": bbox_stats[3] if len(bbox_stats) > 3 else 0,
        "bbox_ap_medium": bbox_stats[4] if len(bbox_stats) > 4 else 0,
        "bbox_ap_large": bbox_stats[5] if len(bbox_stats) > 5 else 0,
        "bbox_ar_max1": bbox_stats[6] if len(bbox_stats) > 6 else 0,
        "bbox_ar_max10": bbox_stats[7] if len(bbox_stats) > 7 else 0,
        "bbox_ar_max100": bbox_stats[8] if len(bbox_stats) > 8 else 0,
        "segm_map50_95": segm_stats[0] if len(segm_stats) > 0 else 0,
        "segm_map50": segm_stats[1] if len(segm_stats) > 1 else 0,
        "segm_map75": segm_stats[2] if len(segm_stats) > 2 else 0,
        "segm_ap_small": segm_stats[3] if len(segm_stats) > 3 else 0,
        "segm_ap_medium": segm_stats[4] if len(segm_stats) > 4 else 0,
        "segm_ap_large": segm_stats[5] if len(segm_stats) > 5 else 0,
        "segm_ar_max1": segm_stats[6] if len(segm_stats) > 6 else 0,
        "segm_ar_max10": segm_stats[7] if len(segm_stats) > 7 else 0,
        "segm_ar_max100": segm_stats[8] if len(segm_stats) > 8 else 0,
    }
    
    with open(save_path, "r", encoding="utf-8") as f:
        result_content = f.read()
    
    return {"success": True, "result": result_content, "data": evaluation_data, "save_path": save_path}

@router.get("/api/evaluations")
def get_evaluations(username: str = None, password: str = None):
    """获取用户评估结果列表"""
    if username and password:
        from server.main import user_manager, file_manager
        success, message = user_manager.login_user(username, password)
        if not success:
            return JSONResponse(status_code=401, content={"error": message})
        
        user_id = user_manager.get_user_id(username)
        evaluations = file_manager.get_user_evaluations(user_id)
        return {"success": True, "evaluations": evaluations}
    return {"success": False, "error": "缺少认证信息"}

@router.delete("/api/evaluations/{eval_id}")
def delete_evaluation(eval_id: int, username: str = None, password: str = None):
    """删除评估记录"""
    if username and password:
        from server.main import user_manager, file_manager
        success, message = user_manager.login_user(username, password)
        if not success:
            return JSONResponse(status_code=401, content={"error": message})
        
        user_id = user_manager.get_user_id(username)
        file_manager.delete_evaluation(eval_id, user_id)
        return {"success": True, "message": "评估记录已删除"}
    return {"success": False, "error": "缺少认证信息"}
