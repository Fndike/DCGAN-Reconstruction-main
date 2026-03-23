'''
DCGAN 三维地质模型重建 - 推理脚本

优化版本：支持分批加载数据

功能：
1. 加载训练好的模型
2. 输入剖面数据，生成完整的三维地质模型
3. 保存生成结果
'''

import numpy as np
import os
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob

from net import Generator

parser = argparse.ArgumentParser(description='DCGAN 3D Geological Model Reconstruction Inference')

parser.add_argument("--test_data_dir", default='./dataset/processed/', help="Directory containing test data files")
parser.add_argument("--model_path", default='./model_save/', help="Path to trained model")
parser.add_argument("--output_dir", default='./test_output/', help="Path to save output results")
parser.add_argument("--image_size", type=int, default=64, help="Image size (width and height)")
parser.add_argument("--image_size_z", type=int, default=64, help="Image size (depth)")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")

args = parser.parse_args()


def get_test_files(data_dir):
    """获取测试数据文件列表"""
    data_files = sorted(glob.glob(os.path.join(data_dir, "test_data_*.npz")))
    label_files = sorted(glob.glob(os.path.join(data_dir, "test_labels_*.npz")))
    return list(zip(data_files, label_files))


def load_test_data(data_path, label_path):
    """加载测试数据"""
    data = np.load(data_path)['data']
    labels = np.load(label_path)['data']
    
    if len(data.shape) == 4:
        data = np.expand_dims(data, axis=-1)
        data = np.concatenate([data] * 3, axis=-1)
    if len(labels.shape) == 4:
        labels = np.expand_dims(labels, axis=-1)
        labels = np.concatenate([labels] * 3, axis=-1)
    
    return data, labels


def save_result(profile, label, generated, output_path):
    """保存生成结果"""
    profile = (profile + 1) / 2 * 255
    label = (label + 1) / 2 * 255
    generated = (generated + 1) / 2 * 255
    
    np.savez_compressed(output_path, 
                        profile=profile.astype(np.int32), 
                        label=label.astype(np.int32), 
                        generated=generated.astype(np.int32))


def visualize_result(profile, label, generated, output_path):
    """可视化结果"""
    import matplotlib.pyplot as plt
    
    profile_2d = profile[:, :, profile.shape[2]//2]
    label_2d = label[:, :, label.shape[2]//2]
    generated_2d = generated[:, :, generated.shape[2]//2]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(profile_2d, cmap='jet')
    axes[0].set_title("Input Profile (Mid-slice)")
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(label_2d, cmap='jet')
    axes[1].set_title("Ground Truth (Mid-slice)")
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(generated_2d, cmap='jet')
    axes[2].set_title("Generated (Mid-slice)")
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def inference():
    """主推理函数"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    test_file_pairs = get_test_files(args.test_data_dir)
    print(f"[INFO] 找到 {len(test_file_pairs)} 个测试数据文件")
    
    test_data_ph = tf.placeholder(
        tf.float32, 
        shape=[None, args.image_size, args.image_size, args.image_size_z, 3],
        name='test_data'
    )
    
    print("[INFO] 构建生成器...")
    gen_output = Generator(image_3D=test_data_ph, gf_dim=64, reuse=False, name='generator')
    
    restore_vars = [v for v in tf.global_variables() if 'generator' in v.name]
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    saver = tf.train.Saver(var_list=restore_vars)
    
    checkpoint = tf.train.latest_checkpoint(args.model_path)
    if checkpoint is None:
        raise FileNotFoundError(f"未找到模型文件: {args.model_path}")
    
    print(f"[INFO] 加载模型: {checkpoint}")
    saver.restore(sess, checkpoint)
    
    print(f"[INFO] 开始推理...")
    
    sample_idx = 0
    
    for data_path, label_path in test_file_pairs:
        print(f"[INFO] 处理: {os.path.basename(data_path)}")
        
        test_data, test_labels = load_test_data(data_path, label_path)
        num_samples = test_data.shape[0]
        
        for i in range(0, num_samples, args.batch_size):
            batch_data = test_data[i:i+args.batch_size]
            batch_labels = test_labels[i:i+args.batch_size]
            
            gen_val = sess.run(gen_output, feed_dict={test_data_ph: batch_data})
            
            for j in range(len(batch_data)):
                output_path = os.path.join(args.output_dir, f'result_{sample_idx:04d}.npz')
                save_result(batch_data[j], batch_labels[j], gen_val[j], output_path)
                
                vis_path = os.path.join(args.output_dir, f'result_{sample_idx:04d}.png')
                visualize_result(batch_data[j], batch_labels[j], gen_val[j], vis_path)
                
                sample_idx += 1
        
        print(f"[INFO] 已处理 {sample_idx} 个样本")
    
    print(f"[INFO] 推理完成! 结果保存在: {args.output_dir}")
    
    sess.close()


if __name__ == "__main__":
    inference()
