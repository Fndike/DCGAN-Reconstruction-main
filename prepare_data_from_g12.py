'''
将 .g12.gz 格式的三维地质模型数据转换为 DCGAN 训练所需的格式

修复版本：正确处理剖面数据的填充值

功能：
1. 解压并读取 .g12.gz 文件
2. 将 200×200×200 的大模型切割成 64×64×64 的小数据块
3. 从完整模型中提取剖面数据作为训练输入
4. 分批保存为 .npz 格式供训练使用
'''

import gzip
import shutil
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Process g12.gz files for DCGAN training')

parser.add_argument("--input_dir", default="./dataset/gz_file/", help="Input directory containing .g12.gz files")
parser.add_argument("--output_dir", default="./dataset/processed/", help="Output directory for processed data")
parser.add_argument("--cube_size", type=int, default=64, help="Size of output cubes (64x64x64)")
parser.add_argument("--stride", type=int, default=64, help="Stride for sliding window (no overlap if equal to cube_size)")
parser.add_argument("--profile_axis", default="x", choices=["x", "y", "z", "xy", "xz", "yz", "xyz"], help="Which axes to extract profiles from")
parser.add_argument("--profile_spacing", type=int, default=8, help="Spacing between profile slices")
parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")
parser.add_argument("--batch_save_size", type=int, default=500, help="Number of samples per npz file")
parser.add_argument("--min_unique_labels", type=int, default=2, help="Minimum unique values required in label block")

args = parser.parse_args()
np.random.seed(1234)

GLOBAL_MIN = 1
GLOBAL_MAX = 8
FILL_VALUE = 0.0


def decompress_g12gz(gz_path):
    """解压 .g12.gz 文件，返回解压后的文件路径"""
    out_path = gz_path[:-3]
    if not os.path.exists(out_path):
        print(f"[INFO] 解压: {gz_path}")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    return out_path


def read_g12_file(g12_path):
    """读取 .g12 文件，返回 200×200×200 的三维数组"""
    print(f"[INFO] 读取: {g12_path}")
    raw_data = np.loadtxt(g12_path, dtype=np.int32)
    vol_3d = raw_data.reshape((200, 200, 200))
    return vol_3d


def extract_cubes_generator(vol_3d, cube_size=64, stride=64):
    """
    生成器：从大体积数据中提取小立方体块（避免内存累积）
    """
    x_max = vol_3d.shape[0] - cube_size + 1
    y_max = vol_3d.shape[1] - cube_size + 1
    z_max = vol_3d.shape[2] - cube_size + 1
    
    for x in range(0, x_max, stride):
        for y in range(0, y_max, stride):
            for z in range(0, z_max, stride):
                cube = vol_3d[x:x+cube_size, y:y+cube_size, z:z+cube_size]
                yield cube


def normalize_data_global(data, global_min=GLOBAL_MIN, global_max=GLOBAL_MAX):
    """
    使用全局最小/最大值归一化数据到 [-1, 1] 范围
    
    原始数据值范围是 1~8 (预留空间给后续新数据)
    归一化后: 
      1 -> -1.000, 2 -> -0.714, 3 -> -0.429, 4 -> -0.143
      5 -> 0.143, 6 -> 0.429, 7 -> 0.714, 8 -> 1.000
    
    注意：填充值 0 在此函数之前已经处理，不会被归一化
    """
    data = data.astype(np.float32)
    data = 2.0 * (data - global_min) / (global_max - global_min) - 1.0
    return data


def extract_profiles_normalized(cube_3d):
    """
    从完整三维模型中提取剖面数据并归一化
    
    每个方向提取2个剖面:
    - X方向: X=16, X=48
    - Y方向: Y=16, Y=48
    - Z方向: Z=16, Z=48
    
    这样确保：
    - 剖面位置：归一化后的真实值
    - 非剖面位置：填充值 FILL_VALUE (0.0)
    """
    cube_norm = normalize_data_global(cube_3d)
    
    profile_data = np.full(cube_3d.shape, FILL_VALUE, dtype=np.float32)
    
    size = cube_3d.shape[0]
    pos1 = size // 4       # 16 (for size=64)
    pos2 = 3 * size // 4   # 48 (for size=64)
    
    profile_data[pos1, :, :] = cube_norm[pos1, :, :]
    profile_data[pos2, :, :] = cube_norm[pos2, :, :]
    
    profile_data[:, pos1, :] = cube_norm[:, pos1, :]
    profile_data[:, pos2, :] = cube_norm[:, pos2, :]
    
    profile_data[:, :, pos1] = cube_norm[:, :, pos1]
    profile_data[:, :, pos2] = cube_norm[:, :, pos2]
    
    return profile_data, cube_norm


def save_batch(data_list, label_list, output_dir, prefix, batch_idx):
    """保存一批数据到npz文件"""
    data_arr = np.array(data_list)
    label_arr = np.array(label_list)
    
    data_path = os.path.join(output_dir, f"{prefix}_data_{batch_idx:04d}.npz")
    label_path = os.path.join(output_dir, f"{prefix}_labels_{batch_idx:04d}.npz")
    
    np.savez_compressed(data_path, data=data_arr)
    np.savez_compressed(label_path, data=label_arr)
    
    return len(data_list)


def process_all_files(input_dir, output_dir, cube_size=64, stride=64, 
                      profile_axis="x", profile_spacing=8, train_ratio=0.8,
                      batch_save_size=500, min_unique_labels=2):
    """
    处理所有 .g12.gz 文件，分批保存
    """
    os.makedirs(output_dir, exist_ok=True)
    
    gz_files = [f for f in os.listdir(input_dir) if f.endswith('.g12.gz')]
    print(f"[INFO] 找到 {len(gz_files)} 个 .g12.gz 文件")
    print(f"[INFO] 全局归一化: min={GLOBAL_MIN}, max={GLOBAL_MAX}")
    print(f"[INFO] 填充值: {FILL_VALUE}")
    
    train_batch_idx = 0
    test_batch_idx = 0
    
    current_train_data = []
    current_train_labels = []
    current_test_data = []
    current_test_labels = []
    
    total_train_samples = 0
    total_test_samples = 0
    total_candidate_patches = 0
    skipped_constant_label_patches = 0
    skipped_low_variation_label_patches = 0
    skipped_constant_profile_patches = 0
    saved_patches = 0
    
    for gz_file in gz_files:
        print(f"\n[INFO] 处理文件: {gz_file}")
        gz_path = os.path.join(input_dir, gz_file)
        
        g12_path = decompress_g12gz(gz_path)
        vol_3d = read_g12_file(g12_path)
        
        cube_count = 0
        for cube in extract_cubes_generator(vol_3d, cube_size=cube_size, stride=stride):
            cube_count += 1
            total_candidate_patches += 1

            label_block = cube
            if label_block.min() == label_block.max():
                skipped_constant_label_patches += 1
                continue

            if np.unique(label_block).size < max(2, min_unique_labels):
                skipped_low_variation_label_patches += 1
                continue
            
            profile_norm, cube_norm = extract_profiles_normalized(cube)

            if np.unique(profile_norm).size < 2:
                skipped_constant_profile_patches += 1
                continue

            saved_patches += 1
            
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
        
        print(f"[INFO] 从 {gz_file} 提取了 {cube_count} 个立方体")
    
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
    
    print(f"\n{'='*50}")
    print(f"total candidate patches: {total_candidate_patches}")
    print(f"skipped constant label patches: {skipped_constant_label_patches}")
    print(f"skipped low-variation label patches: {skipped_low_variation_label_patches}")
    print(f"skipped constant profile patches: {skipped_constant_profile_patches}")
    print(f"saved patches: {saved_patches}")
    print(f"[SUCCESS] 数据处理完成!")
    print(f"  训练样本总数: {total_train_samples}")
    print(f"  测试样本总数: {total_test_samples}")
    print(f"  训练数据文件: {train_batch_idx + 1} 个")
    print(f"  测试数据文件: {test_batch_idx + 1} 个")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*50}")
    
    with open(os.path.join(output_dir, "dataset_info.txt"), "w") as f:
        f.write(f"train_samples: {total_train_samples}\n")
        f.write(f"test_samples: {total_test_samples}\n")
        f.write(f"train_batches: {train_batch_idx + 1}\n")
        f.write(f"test_batches: {test_batch_idx + 1}\n")
        f.write(f"cube_size: {cube_size}\n")
        f.write(f"profile_axis: {profile_axis}\n")
        f.write(f"profile_spacing: {profile_spacing}\n")
        f.write(f"global_min: {GLOBAL_MIN}\n")
        f.write(f"global_max: {GLOBAL_MAX}\n")
        f.write(f"fill_value: {FILL_VALUE}\n")


if __name__ == "__main__":
    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        cube_size=args.cube_size,
        stride=args.stride,
        profile_axis=args.profile_axis,
        profile_spacing=args.profile_spacing,
        train_ratio=args.train_ratio,
        batch_save_size=args.batch_save_size,
        min_unique_labels=args.min_unique_labels
    )
