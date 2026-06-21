import numpy as np
import os
from scipy.ndimage import label

def check_dataset_safety(data_dir, merge_tol=2.0):
    print("================ 开始全库空间安全筛查 ================")
    files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    total_samples = 0
    total_warnings = 0
    
    # 抽样或全量检查前100个样本（可以自行调整）
    for file_name in files[:100]:
        file_path = os.path.join(data_dir, file_name)
        data = np.load(file_path)
        # 假设读取出来的原始三维矩阵叫 cube_3d
        cube_3d = data['cube_3d'] if 'cube_3d' in data else data[data.files[0]]
        cube_3d = np.squeeze(cube_3d)
        
        unique_vals = np.sort(np.unique(cube_3d))
        total_samples += 1
        
        for i in range(len(unique_vals) - 1):
            v1 = unique_vals[i]
            v2 = unique_vals[i+1]
            
            # 如果数值靠得很近，触发空间审查
            if v2 - v1 <= merge_tol:
                mask_v1 = (cube_3d == v1)
                mask_v2 = (cube_3d == v2)
                mask_combined = mask_v1 | mask_v2
                
                # 计算它们各自和合并后的独立空间连通域数量
                _, num_v1 = label(mask_v1)
                _, num_v2 = label(mask_v2)
                _, num_combined = label(mask_combined)
                
                # 【黄金数学判据】：
                # 如果它们在空间上互不接触，合并后的连通域数量必然等于各自的数量之和。
                if num_combined == (num_v1 + num_v2):
                    print(f"⚠️ [发现误伤] 文件 {file_name}: 数值 {v1} 和 {v2} 极近(差{v2-v1:.4f})，但在空间上完全隔离！")
                    total_warnings += 1
                    
    print("--------------------------------------------------")
    print(f"筛查结束：共检查 {total_samples} 个样本，共发现 {total_warnings} 处空间隔离误伤。")

# 填入你师兄的原始未清洗数据集路径
check_dataset_safety('dataset\profile_data')