import torch
import numpy as np
import cv2
import os
import argparse
from glob import glob
from custom_mst_plus_plus import CustomMSTPlusPlus
from tqdm import tqdm
import datetime

def load_model(model_path):
    """加载训练好的模型"""
    model = CustomMSTPlusPlus(in_channels=3, out_channels=7, n_feat=31, stage=3)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 处理可能的DataParallel包装
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:
        # 创建新的state_dict，移除'module.'前缀
        new_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(model, input_dir, output_dir, device):
    """处理单个场景的图像"""
    # 获取输入文件（3通道）
    input_files = sorted(glob(os.path.join(input_dir, "*.bmp")))
    
    if len(input_files) != 3:
        print(f"警告: 目录 {input_dir} 中的通道数量不正确。预期3个输入通道，实际获得 {len(input_files)} 个。")
        return
    
    # 读取输入图像
    input_channels = []
    for input_file in input_files:
        img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
        input_channels.append(img)
    
    # 堆叠输入通道
    input_data = np.stack(input_channels, axis=0)  # 形状: [3, H, W]
    input_data = input_data.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    
    # 转换为PyTorch张量并移动到设备
    input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(device)  # 形状: [1, 3, H, W]
    
    # 推理
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # 将输出转换回NumPy数组
    output_data = output_tensor.squeeze(0).cpu().numpy()  # 形状: [7, H, W]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每个输出通道为单独的BMP文件
    for i, channel in enumerate(output_data):
        # 将值缩放回0-255范围
        channel_scaled = (channel * 255.0).clip(0, 255).astype(np.uint8)
        output_path = os.path.join(output_dir, f"{i+1}.bmp")
        cv2.imwrite(output_path, channel_scaled)
        print(f"已保存通道 {i+1} 到 {output_path}")

def main():
    parser = argparse.ArgumentParser(description="使用MST++模型进行三通道到七通道的推理")
    parser.add_argument("--model_dir", type=str, default="./exp/custom_mst_plus_plus/2025_03_08_16_10_07/", 
                        help="包含模型检查点的目录")
    parser.add_argument("--model_path", type=str, default=None,
                        help="特定模型检查点的路径，如果提供，将覆盖model_dir")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="包含输入场景的目录")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="保存输出结果的目录")
    parser.add_argument("--gpu_id", type=str, default="0", 
                        help="要使用的GPU ID")
    parser.add_argument("--select_model", type=str, default="latest", 
                        choices=["latest", "best"],
                        help="选择模型的策略: 'latest'表示最新的模型, 'best'表示验证损失最低的模型")
    args = parser.parse_args()
    
    # 设置GPU
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 确定要使用的模型路径
    if args.model_path is not None:
        model_path = args.model_path
        print(f"使用指定的模型: {model_path}")
    else:
        model_files = glob(os.path.join(args.model_dir, "model_*.pth"))
        if not model_files:
            print(f"错误: 在 {args.model_dir} 中未找到模型检查点")
            return
        
        if args.select_model == "latest":
            model_path = sorted(model_files, key=lambda x: int(x.split('_iter_')[1].split('.pth')[0]))[-1]
            print(f"使用最新的模型 (最高迭代次数): {model_path}")
        else:
            best_model = os.path.join(args.model_dir, "best_model.pth")
            if os.path.exists(best_model):
                model_path = best_model
                print(f"使用最佳模型: {model_path}")
            else:
                model_path = sorted(model_files, key=lambda x: int(x.split('_iter_')[1].split('.pth')[0]))[-1]
                print(f"未找到明确标记的最佳模型，使用最新的模型: {model_path}")
    
    # 加载模型
    model = load_model(model_path)
    model.to(device)
    model.eval()

    start_time = datetime.datetime.now()

    # 检查输入目录是否包含子目录（多个场景）
    scene_dirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]

    if scene_dirs:
        for scene_dir in tqdm(scene_dirs, desc="处理场景"):
            input_scene_path = os.path.join(args.input_dir, scene_dir)
            output_scene_path = os.path.join(args.output_dir, scene_dir)
            process_image(model, input_scene_path, output_scene_path, device)
    else:
        process_image(model, args.input_dir, args.output_dir, device)

    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time

    # 打印时间信息
    print("推理完成!")
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed_time}")

    # 记录日志文件
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "inference_time_log.txt"), "w") as log_file:
        log_file.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"总耗时: {elapsed_time}\n")

if __name__ == "__main__":
    main()

