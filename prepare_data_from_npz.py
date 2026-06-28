'''
将批量电阻率 .npz 数据转换为 DCGAN 训练所需的格式

适配数据集：形状为 [N, 64, 64, 32] 的电阻率 NPZ 文件

功能：
1. 直接读取 .npz 文件（内部键为 'data'），float16→float32 无损转换
2. 对每个样本进行局部阶梯归一化（动态映射到 [-1, 1]）
3. 提取多方向剖面数据作为条件输入（Fence 测线网格：X[16,32,48] Y[16,32,48] Z[8,24]）
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

FILL_VALUE = -2.0


def normalize_staircase(cube_3d, max_classes=16):
    """
    全库绝对物理标度对齐：
    不管样本实际有几层，先 KMeans 聚类到固定的 max_classes 类，
    再按 max_classes 个均匀阶梯映射到 [-1, 1]。
    整个数据集中所有样本共享相同的 16 把尺子刻度。

    返回: (归一化矩阵, 排序后的独特值数组, 实际类别数 K)
    """
    flat = cube_3d.reshape(-1, 1)
    K_local = len(np.unique(cube_3d))

    if K_local <= max_classes:
        # 类别数 ≤ 16：直接线性映射到 max_classes 阶梯
        unique_vals = np.sort(np.unique(cube_3d))
        id_map = np.searchsorted(unique_vals, cube_3d).astype(np.float32)
        K = K_local
    else:
        # 类别数 > 16：用 KMeans 聚类压缩到 max_classes 类
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=max_classes, random_state=42, n_init=10)
        id_map = km.fit_predict(flat).reshape(cube_3d.shape).astype(np.float32)
        K = max_classes

    # 【核心】：永远除以固定的 (max_classes - 1)，刻度绝对统一！
    cube_norm = (id_map / (max_classes - 1)) * 2.0 - 1.0
    return cube_norm, id_map, K


def extract_profiles_normalized(cube_3d, cube_norm, K):
    """
    从完整三维模型中提取剖面数据并归一化

    模型形状: (64, 64, 32)
    - X轴 (len=64): 切片位置 16, 32, 48
    - Y轴 (len=64): 切片位置 16, 32, 48
    - Z轴 (len=32): 切片位置 8, 24

    非剖面位置填充 FILL_VALUE，剖面位置直接复用 cube_norm 的归一化值。
    cube_norm 是 normalize_staircase 返回的、与训练空间刻度一致的归一化矩阵。
    """
    profile_raw = np.full(cube_3d.shape, FILL_VALUE, dtype=np.float32)

    # X轴切片（3张）：Fence 测线网格索引 16, 32, 48
    profile_raw[16, :, :] = cube_3d[16, :, :]
    profile_raw[32, :, :] = cube_3d[32, :, :]
    profile_raw[48, :, :] = cube_3d[48, :, :]
    # Y轴切片（3张）：Fence 测线网格索引 16, 32, 48
    profile_raw[:, 16, :] = cube_3d[:, 16, :]
    profile_raw[:, 32, :] = cube_3d[:, 32, :]
    profile_raw[:, 48, :] = cube_3d[:, 48, :]
    # Z轴切片（2张）：Fence 测线网格索引 8, 24
    profile_raw[:, :, 8] = cube_3d[:, :, 8]
    profile_raw[:, :, 24] = cube_3d[:, :, 24]

    # 二值掩码：剖面位置为1.0，未知盲区为0.0
    mask = (profile_raw != FILL_VALUE).astype(np.float32)
    # 数据通道：盲区保持 FILL_VALUE，剖面位置直接用 cube_norm 的归一化值
    profile_norm = np.full(cube_3d.shape, FILL_VALUE, dtype=np.float32)
    profile_norm[mask > 0.5] = cube_norm[mask > 0.5]

    # 合并为双通道：[数据通道, 掩码通道]
    profile_dual = np.stack([profile_norm, mask], axis=-1)
    return profile_dual


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
    print(f"[INFO] 剖面切片: Fence 测线网格 X[16,32,48] Y[16,32,48] Z[8,24]")
    
    train_batch_idx = 0
    test_batch_idx = 0
    
    current_train_data = []
    current_train_labels = []
    current_test_data = []
    current_test_labels = []
    
    total_train_samples = 0
    total_test_samples = 0
    total_samples = 0
    filtered_samples = 0
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

            # 极简纯离散地层过滤器：只保留 K ∈ [5, 16] 的无渐变纯断块模型
            unique_vals_raw = np.unique(cube_3d)
            K_raw = len(unique_vals_raw)
            if K_raw > 16 or K_raw < 5:
                filtered_samples += 1
                continue  # 无条件剔除：渐变场（>16）或退化样本（<5）

            # 阶梯归一化
            cube_norm, unique_vals, K = normalize_staircase(cube_3d)

            # 提取剖面（直接复用 cube_norm 的归一化值，避免重复 searchsorted）
            profile_norm = extract_profiles_normalized(cube_3d, cube_norm, K)
            
            saved_patches += 1
            K_list.append(K)
            label_std_list.append(np.std(cube_norm))
            profile_filled_ratio = np.count_nonzero(profile_norm[..., 1]) / profile_norm[..., 1].size
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
    pass_rate = (1 - filtered_samples / total_samples) * 100 if total_samples > 0 else 0
    print(f"\n{'='*50}")
    print(f"total samples processed: {total_samples}")
    print(f"filtered out (K<5 or K>16): {filtered_samples} ({filtered_samples/total_samples*100:.1f}%)")
    print(f"passed filter: {total_samples - filtered_samples} ({pass_rate:.1f}%)")
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
        f.write(f"profile_slices: X[16,32,48] Y[16,32,48] Z[8,24]\n")


if __name__ == "__main__":
    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        batch_save_size=args.batch_save_size
    )
