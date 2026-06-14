'''
将批量电阻率 .npz 数据转换为 DCGAN 训练所需的格式

适配数据集：形状为 [N, 64, 64, 32] 的电阻率 NPZ 文件

功能：
1. 直接读取 .npz 文件（内部键为 'data'），float16→float32 无损转换
2. 对每个样本进行局部阶梯归一化（动态映射到 [-1, 1]）
3. 提取 Z轴16切片 的多方向剖面数据作为条件输入
4. 分批保存为 .npz 格式供训练使用
'''

import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Process NPZ resistivity data for DCGAN training')

parser.add_argument("--input_dir", default="./dataset/npz_file/", help="Input directory containing .npz files")
parser.add_argument("--output_dir", default="./dataset/processed/", help="Output directory for processed data")
parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")
parser.add_argument("--batch_save_size", type=int, default=500, help="Number of samples per npz file")

args = parser.parse_args()
np.random.seed(1234)

FILL_VALUE = 0.0


def normalize_staircase(cube_3d):
    """
    样本级局部阶梯归一化
    
    对每一个独立的 [64, 64, 32] 模型：
    1. 获取所有独特电阻率值，从小到大排序
    2. 动态映射为 0 至 K-1 的连续阶梯整型 ID
    3. 归一化到 [-1, 1]: X_norm = (X_id / (K-1)) * 2.0 - 1.0
    
    返回: (归一化矩阵, 排序后的独特值数组, 类别数K)
    """
    unique_vals = np.sort(np.unique(cube_3d))
    K = len(unique_vals)
    
    # 使用 searchsorted 将原始值映射为阶梯 ID
    id_map = np.searchsorted(unique_vals, cube_3d).astype(np.float32)
    
    if K <= 1:
        cube_norm = np.zeros_like(cube_3d, dtype=np.float32)
    else:
        cube_norm = (id_map / (K - 1)) * 2.0 - 1.0
    
    return cube_norm, unique_vals, K


def extract_profiles_normalized(cube_3d, unique_vals, K):
    """
    从完整三维模型中提取剖面数据并归一化
    
    模型形状: (64, 64, 32)
    - X轴 (len=64): 切片位置 16, 48
    - Y轴 (len=64): 切片位置 16, 48
    - Z轴 (len=32): 切片位置 16 (仅此一处，48会越界)
    
    非剖面位置填充 0.0，剖面位置使用与 label 相同的阶梯归一化
    """
    profile_raw = np.full(cube_3d.shape, FILL_VALUE, dtype=np.float32)
    
    # X轴切片
    profile_raw[16, :, :] = cube_3d[16, :, :]
    profile_raw[48, :, :] = cube_3d[48, :, :]
    # Y轴切片
    profile_raw[:, 16, :] = cube_3d[:, 16, :]
    profile_raw[:, 48, :] = cube_3d[:, 48, :]
    # Z轴切片 (仅在16处)
    profile_raw[:, :, 16] = cube_3d[:, :, 16]
    
    # 只对非填充位置做阶梯归一化，填充位置保持 0.0
    mask = profile_raw != FILL_VALUE
    profile_norm = np.full(cube_3d.shape, FILL_VALUE, dtype=np.float32)
    
    if K > 1 and mask.any():
        raw_values = profile_raw[mask]
        ids = np.searchsorted(unique_vals, raw_values).astype(np.float32)
        profile_norm[mask] = (ids / (K - 1)) * 2.0 - 1.0
    elif K == 1 and mask.any():
        profile_norm[mask] = 0.0
    
    return profile_norm


def save_batch(data_list, label_list, output_dir, prefix, batch_idx):
    """保存一批数据到npz文件"""
    data_arr = np.array(data_list)
    label_arr = np.array(label_list)
    
    data_path = os.path.join(output_dir, f"{prefix}_data_{batch_idx:04d}.npz")
    label_path = os.path.join(output_dir, f"{prefix}_labels_{batch_idx:04d}.npz")
    
    np.savez_compressed(data_path, data=data_arr)
    np.savez_compressed(label_path, data=label_arr)
    
    return len(data_list)


def process_all_files(input_dir, output_dir, train_ratio=0.8, batch_save_size=500):
    """
    处理所有 .npz 文件，分批保存
    """
    os.makedirs(output_dir, exist_ok=True)
    
    npz_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npz')])
    print(f"[INFO] 找到 {len(npz_files)} 个 .npz 文件")
    print(f"[INFO] 填充值: {FILL_VALUE}")
    print(f"[INFO] 归一化方式: 样本级局部阶梯归一化")
    print(f"[INFO] 剖面切片: X[16,48] Y[16,48] Z[16]")
    
    train_batch_idx = 0
    test_batch_idx = 0
    
    current_train_data = []
    current_train_labels = []
    current_test_data = []
    current_test_labels = []
    
    total_train_samples = 0
    total_test_samples = 0
    total_samples = 0
    saved_patches = 0
    K_list = []
    label_std_list = []
    profile_filled_ratio_list = []
    
    for npz_file in npz_files:
        print(f"\n[INFO] 处理文件: {npz_file}")
        npz_path = os.path.join(input_dir, npz_file)
        
        raw = np.load(npz_path)['data']  # shape: [N, 64, 64, 32]
        raw = raw.astype(np.float32)
        
        print(f"[INFO] 数据形状: {raw.shape}, 数据类型: {raw.dtype}")
        
        for i in range(raw.shape[0]):
            cube_3d = raw[i]  # (64, 64, 32)
            total_samples += 1
            
            # 阶梯归一化
            cube_norm, unique_vals, K = normalize_staircase(cube_3d)
            
            # 提取剖面
            profile_norm = extract_profiles_normalized(cube_3d, unique_vals, K)
            
            saved_patches += 1
            K_list.append(K)
            label_std_list.append(np.std(cube_norm))
            profile_filled_ratio = np.count_nonzero(profile_norm != FILL_VALUE) / profile_norm.size
            profile_filled_ratio_list.append(profile_filled_ratio)
            
            if np.random.random() < train_ratio:
                current_train_data.append(profile_norm)
                current_train_labels.append(cube_norm)
                
                if len(current_train_data) >= batch_save_size:
                    saved = save_batch(current_train_data, current_train_labels,
                                      output_dir, "train", train_batch_idx)
                    total_train_samples += saved
                    print(f"[INFO] 保存训练批次 {train_batch_idx}: {saved} 样本")
                    train_batch_idx += 1
                    current_train_data = []
                    current_train_labels = []
            else:
                current_test_data.append(profile_norm)
                current_test_labels.append(cube_norm)
                
                if len(current_test_data) >= batch_save_size:
                    saved = save_batch(current_test_data, current_test_labels,
                                      output_dir, "test", test_batch_idx)
                    total_test_samples += saved
                    print(f"[INFO] 保存测试批次 {test_batch_idx}: {saved} 样本")
                    test_batch_idx += 1
                    current_test_data = []
                    current_test_labels = []
        
        print(f"[INFO] 从 {npz_file} 处理了 {raw.shape[0]} 个样本")
    
    # 保存剩余数据
    if current_train_data:
        saved = save_batch(current_train_data, current_train_labels,
                          output_dir, "train", train_batch_idx)
        total_train_samples += saved
        print(f"[INFO] 保存最后训练批次 {train_batch_idx}: {saved} 样本")
    
    if current_test_data:
        saved = save_batch(current_test_data, current_test_labels,
                          output_dir, "test", test_batch_idx)
        total_test_samples += saved
        print(f"[INFO] 保存最后测试批次 {test_batch_idx}: {saved} 样本")
    
    # 统计信息
    print(f"\n{'='*50}")
    print(f"total samples processed: {total_samples}")
    print(f"saved patches: {saved_patches}")
    if K_list:
        k = np.array(K_list)
        print("[INFO] K (unique values per sample) stats:")
        print(f"    min={k.min()}  p10={np.percentile(k,10):.1f}  p25={np.percentile(k,25):.1f}  "
              f"p50={np.percentile(k,50):.1f}  p75={np.percentile(k,75):.1f}  p90={np.percentile(k,90):.1f}  max={k.max()}")
    if label_std_list:
        s = np.array(label_std_list)
        print("[INFO] label cube_norm std stats (saved patches only):")
        print(f"    min={s.min():.6f}  p10={np.percentile(s,10):.6f}  p25={np.percentile(s,25):.6f}  "
              f"p50={np.percentile(s,50):.6f}  p75={np.percentile(s,75):.6f}  p90={np.percentile(s,90):.6f}  max={s.max():.6f}")
    if profile_filled_ratio_list:
        r = np.array(profile_filled_ratio_list)
        print("[INFO] profile filled ratio stats (saved patches only):")
        print(f"    min={r.min():.6f}  p10={np.percentile(r,10):.6f}  p25={np.percentile(r,25):.6f}  "
              f"p50={np.percentile(r,50):.6f}  p75={np.percentile(r,75):.6f}  p90={np.percentile(r,90):.6f}  max={r.max():.6f}")
    print(f"[SUCCESS] 数据处理完成!")
    print(f"  训练样本总数: {total_train_samples}")
    print(f"  测试样本总数: {total_test_samples}")
    print(f"  训练数据文件: {train_batch_idx + (1 if current_train_data else 0)} 个")
    print(f"  测试数据文件: {test_batch_idx + (1 if current_test_data else 0)} 个")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*50}")
    
    with open(os.path.join(output_dir, "dataset_info.txt"), "w") as f:
        f.write(f"train_samples: {total_train_samples}\n")
        f.write(f"test_samples: {total_test_samples}\n")
        f.write(f"normalization: staircase_per_sample\n")
        f.write(f"fill_value: {FILL_VALUE}\n")
        f.write(f"profile_slices: X[16,48] Y[16,48] Z[16]\n")


if __name__ == "__main__":
    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        batch_save_size=args.batch_save_size
    )
