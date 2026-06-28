'''
DCGAN 三维地质模型重建 - 训练脚本

优化版本：支持分批加载数据，避免内存溢出

功能：
1. 分批加载预处理好的剖面数据和完整模型数据
2. 训练 DCGAN 网络
3. 保存训练模型和生成结果
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import sys
import numpy as np
import os
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob
import random

from net import Generator, Discriminator

parser = argparse.ArgumentParser(description='DCGAN 3D Geological Model Reconstruction Training')

parser.add_argument("--train_data_dir", default='./dataset/processed/', help="Directory containing training data files")
parser.add_argument("--snapshot_dir", default='./model_save', help='Path to save model checkpoints')
parser.add_argument("--out_dir", default='./train_out', help='Path to save training outputs')
parser.add_argument("--image_size", type=int, default=64, help="Image size (width and height)")
parser.add_argument("--image_size_z", type=int, default=32, help="Image size (depth)")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
parser.add_argument("--epoch", type=int, default=200, help="Number of training epochs")
parser.add_argument("--base_lr_g", type=float, default=0.0002, help="Learning rate for generator")
parser.add_argument("--base_lr_d", type=float, default=0.00002, help="Learning rate for discriminator")
parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
parser.add_argument("--random_seed", type=int, default=1234, help="Random seed")
parser.add_argument("--save_pred_every", type=int, default=1000, help="Save model every N steps")
parser.add_argument("--summary_pred_every", type=int, default=100, help="Save summary every N steps")
parser.add_argument("--write_pred_every", type=int, default=500, help="Write prediction every N steps")
parser.add_argument("--lambda_l1", type=float, default=100.0, help="L1 loss weight")
parser.add_argument("--lambda_gan", type=float, default=1.0, help="GAN loss weight")
parser.add_argument("--lambda_cos", type=float, default=10.0, help="Cosine similarity loss weight")
parser.add_argument("--lambda_tv", type=float, default=0.1, help="3D Total Variation loss weight")
parser.add_argument("--df_dim", type=int, default=32, help="Discriminator feature dimension")

args = parser.parse_args()
EPS = 1e-12


class Logger(object):
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()


def create_directories():
    """创建必要的目录"""
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)


def get_data_files(data_dir, prefix="train"):
    """获取数据文件列表"""
    data_files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_data_*.npz")))
    label_files = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_labels_*.npz")))
    return list(zip(data_files, label_files))


def load_batch_data(data_path, label_path):
    """加载一批数据"""
    data = np.load(data_path)['data']
    labels = np.load(label_path)['data']
    
    # 数据已是 5D (N,64,64,32,2)，无需扩展
    # 标签是 4D (N,64,64,32)，扩展为 5D (N,64,64,32,1)
    if len(labels.shape) == 4:
        labels = np.expand_dims(labels, axis=-1)
    
    return data, labels


class DataGenerator:
    """数据生成器，支持分批加载"""
    
    def __init__(self, data_dir, batch_size, prefix="train"):
        self.file_pairs = get_data_files(data_dir, prefix)
        self.batch_size = batch_size
        self.current_data = None
        self.current_labels = None
        self.current_idx = 0
        self.current_file_idx = 0
        
        print(f"[INFO] 找到 {len(self.file_pairs)} 个数据文件")
    
    def __len__(self):
        total_samples = 0
        for data_path, _ in self.file_pairs:
            data = np.load(data_path)['data']
            total_samples += len(data)
        return total_samples
    
    def reset(self):
        """重置数据生成器"""
        random.shuffle(self.file_pairs)
        self.current_file_idx = 0
        self.current_idx = 0
        self.current_data = None
        self.current_labels = None
    
    def load_next_file(self):
        """加载下一个数据文件"""
        if self.current_file_idx >= len(self.file_pairs):
            return False
        
        data_path, label_path = self.file_pairs[self.current_file_idx]
        self.current_data, self.current_labels = load_batch_data(data_path, label_path)
        
        indices = np.random.permutation(len(self.current_data))
        self.current_data = self.current_data[indices]
        self.current_labels = self.current_labels[indices]
        
        self.current_idx = 0
        self.current_file_idx += 1
        return True
    
    def get_batch(self):
        """获取一个batch的数据"""
        if self.current_data is None or self.current_idx >= len(self.current_data):
            if not self.load_next_file():
                return None, None
        
        batch_data = self.current_data[self.current_idx:self.current_idx + self.batch_size]
        batch_labels = self.current_labels[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        return batch_data, batch_labels


def save_checkpoint(saver, sess, step):
    """保存模型检查点"""
    checkpoint_path = os.path.join(args.snapshot_dir, 'model')
    saver.save(sess, checkpoint_path, global_step=step)
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} [INFO] 模型已保存: step {step}')


def save_prediction(profile, label, generated, output_dir, step):
    """保存预测结果"""
    profile = (profile + 1) / 2 * 255
    label = (label + 1) / 2 * 255
    generated = (generated + 1) / 2 * 255
    
    output_path = os.path.join(output_dir, f'prediction_step_{step}.npz')
    np.savez_compressed(output_path, 
                        profile=profile, 
                        label=label, 
                        generated=generated)


def save_visual_comparison(profile, label, generated, output_dir, step):
    """训练过程中保存三方向中间切片对比图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # profile / label / generated 原始范围都是 [-1, 1]，与训练数据归一化空间一致
    profile = profile[..., 0]
    label   = label[..., 0]
    gen_raw = generated[..., 0]

    # ======= 16 阶梯最近邻吸附：让训练可视化也呈现干净色块（仅 numpy，不进计算图）=======
    global_steps = np.linspace(-1.0, 1.0, 16).astype(np.float32)
    idx = np.argmin(
        np.abs(gen_raw[..., np.newaxis] - global_steps[np.newaxis, np.newaxis, np.newaxis, :]),
        axis=-1
    )
    gen = global_steps[idx]  # 与训练数据完全对齐的 16 阶梯

    vis_z = 16
    vis_y = 24
    vis_x = 24

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # 统一 vmin/vmax 到 [-1, 1]，profile/label 和 generated 共用同一套刻度
    VMIN, VMAX = -1.0, 1.0

    axes[0, 0].imshow(profile[:, :, vis_z], cmap='jet', vmin=VMIN, vmax=VMAX)
    axes[0, 0].set_title("Profile Z=16")
    axes[0, 1].imshow(label[:, :, vis_z], cmap='jet', vmin=VMIN, vmax=VMAX)
    axes[0, 1].set_title("Label Z=16")
    axes[0, 2].imshow(gen[:, :, vis_z], cmap='jet', vmin=VMIN, vmax=VMAX)
    axes[0, 2].set_title("Generated Z=16")

    axes[1, 0].imshow(profile[:, vis_y, :], cmap='jet', vmin=VMIN, vmax=VMAX)
    axes[1, 0].set_title("Profile Y=24")
    axes[1, 1].imshow(label[:, vis_y, :], cmap='jet', vmin=VMIN, vmax=VMAX)
    axes[1, 1].set_title("Label Y=24")
    axes[1, 2].imshow(gen[:, vis_y, :], cmap='jet', vmin=VMIN, vmax=VMAX)
    axes[1, 2].set_title("Generated Y=24")

    axes[2, 0].imshow(profile[vis_x, :, :], cmap='jet', vmin=VMIN, vmax=VMAX)
    axes[2, 0].set_title("Profile X=24")
    axes[2, 1].imshow(label[vis_x, :, :], cmap='jet', vmin=VMIN, vmax=VMAX)
    axes[2, 1].set_title("Label X=24")
    axes[2, 2].imshow(gen[vis_x, :, :], cmap='jet', vmin=VMIN, vmax=VMAX)
    axes[2, 2].set_title("Generated X=24")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"vis_step_{step:06d}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def train():
    """主训练函数"""
    log_file = open('train_log.txt', 'a', encoding='utf-8')
    original_stdout = sys.stdout
    sys.stdout = Logger(sys.stdout, log_file)
    
    tf.set_random_seed(args.random_seed)
    create_directories()
    
    data_gen = DataGenerator(args.train_data_dir, args.batch_size, prefix="train")
    total_samples = len(data_gen)
    
    if total_samples == 0:
        raise ValueError(f"在 {args.train_data_dir} 下没有找到 train_data_*.npz 和 train_labels_*.npz")
    
    steps_per_epoch = total_samples // args.batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    
    print(f"[INFO] 总训练样本数: {total_samples}")
    print(f"[INFO] 每epoch步数: {steps_per_epoch}")
    
    train_data_ph = tf.placeholder(
        tf.float32, 
        shape=[None, args.image_size, args.image_size, args.image_size_z, 2],
        name='train_data'
    )
    train_label_ph = tf.placeholder(
        tf.float32, 
        shape=[None, args.image_size, args.image_size, args.image_size_z, 1],
        name='train_label'
    )

    # 声明 is_training 占位符，控制 Generator 内部 Dropout
    is_training_ph = tf.placeholder(tf.bool, shape=[], name='is_training')

    print("[INFO] 构建生成器...")
    gen_output = Generator(image_3D=train_data_ph, gf_dim=64, reuse=False, is_training=is_training_ph, name='generator')
    
    # ======= 全局步数变量（用于噪声退火、日志、保存）=======
    global_step_var = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
    increment_global_step = tf.assign_add(global_step_var, 1)

    print("[INFO] 构建判别器...")

    # ======= Instance Noise：训练时对送入判别器的标签添加微弱三维高斯噪声 =======
    # 动态噪声退火：0-50k步维持0.05；50k-150k步线性衰减至0；150k步后锁定为0
    noise_level = tf.cond(
        global_step_var < 50000,
        lambda: 0.05,
        lambda: tf.cond(
            global_step_var < 150000,
            lambda: 0.05 * tf.cast(150000 - global_step_var, tf.float32) / 100000.0,
            lambda: 0.0
        )
    )
    noise_real = tf.random.normal(shape=tf.shape(train_label_ph), mean=0.0, stddev=noise_level)
    noise_fake = tf.random.normal(shape=tf.shape(gen_output), mean=0.0, stddev=noise_level)
    train_label_noisy = tf.cond(is_training_ph, lambda: train_label_ph + noise_real, lambda: train_label_ph)
    gen_output_noisy = tf.cond(is_training_ph, lambda: gen_output + noise_fake, lambda: gen_output)

    dis_real = Discriminator(train_data_ph, train_label_noisy, df_dim=args.df_dim, reuse=False, name='discriminator')
    dis_fake = Discriminator(train_data_ph, gen_output_noisy, df_dim=args.df_dim, reuse=True, name='discriminator')
    
    print("[INFO] 计算最小二乘对抗损失（带单侧标签平滑）...")

    # ======= LSGAN 判别器损失（标签平滑：真 0.9 / 假 0.1）=======
    d_loss_real = tf.reduce_mean(tf.square(dis_real - 0.9))
    d_loss_fake = tf.reduce_mean(tf.square(dis_fake - 0.1))
    d_loss = 0.5 * (d_loss_real + d_loss_fake)

    # ======= 生成器对抗损失（LSGAN，目标逼近 0.9）=======
    g_loss_gan = tf.reduce_mean(tf.square(dis_fake - 0.9))

    # ======= L1 重建损失 =======
    g_loss_l1 = tf.reduce_mean(tf.abs(gen_output - train_label_ph))

    # ======= 余弦相似度联合损失 L_cos =======
    dot_product = tf.reduce_sum(gen_output * train_label_ph, axis=[1, 2, 3, 4])
    norm_gen = tf.sqrt(tf.reduce_sum(tf.square(gen_output), axis=[1, 2, 3, 4]))
    norm_label = tf.sqrt(tf.reduce_sum(tf.square(train_label_ph), axis=[1, 2, 3, 4]))
    cosine_similarity = dot_product / (norm_gen * norm_label + EPS)
    g_loss_cos = tf.reduce_mean(1.0 - cosine_similarity)

    # ======= 3D Total Variation Loss =======
    tv_x = tf.reduce_mean(tf.abs(gen_output[:, 1:, :, :, :] - gen_output[:, :-1, :, :, :]))
    tv_y = tf.reduce_mean(tf.abs(gen_output[:, :, 1:, :, :] - gen_output[:, :, :-1, :, :]))
    tv_z = tf.reduce_mean(tf.abs(gen_output[:, :, :, 1:, :] - gen_output[:, :, :, :-1, :]))
    g_loss_tv = tv_x + tv_y + tv_z

    # ======= 生成器总损失 =======
    g_loss = (args.lambda_gan * g_loss_gan
              + args.lambda_l1 * g_loss_l1
              + args.lambda_cos * g_loss_cos
              + args.lambda_tv * g_loss_tv)

    g_loss_sum = tf.summary.scalar('generator_loss', g_loss)
    d_loss_sum = tf.summary.scalar('discriminator_loss', d_loss)
    g_loss_l1_sum = tf.summary.scalar('generator_l1_loss', g_loss_l1)
    g_loss_cos_sum = tf.summary.scalar('generator_cos_loss', g_loss_cos)
    g_loss_tv_sum = tf.summary.scalar('generator_tv_loss', g_loss_tv)
    
    gen_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]
    dis_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]
    
    print(f"[INFO] 生成器参数数量: {sum([np.prod(v.shape) for v in gen_vars])}")
    print(f"[INFO] 判别器参数数量: {sum([np.prod(v.shape) for v in dis_vars])}")
    
    g_optimizer = tf.train.AdamOptimizer(args.base_lr_g, beta1=args.beta1)
    d_optimizer = tf.train.AdamOptimizer(args.base_lr_d, beta1=args.beta1)

    g_train_op = g_optimizer.minimize(g_loss, var_list=gen_vars)
    d_train_op = d_optimizer.minimize(d_loss, var_list=dis_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)
    
    ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        global_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[-1])
        sess.run(global_step_var.assign(global_step))
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} [INFO] 从 {ckpt.model_checkpoint_path} 恢复模型成功，已训练 {global_step} 步')
    else:
        sess.run(tf.global_variables_initializer())
        global_step = 0
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} [INFO] 未找到 checkpoint，从头开始训练')
    
    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())
    
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} [INFO] 开始训练...')

    d_loss_val = 1.0    # 历史状态变量：判别器损失
    g_loss_gan_val = 1.0  # 历史状态变量：生成器对抗损失

    for epoch in range(args.epoch):
        data_gen.reset()

        for step in range(steps_per_epoch):
            global_step += 1
            sess.run(increment_global_step)

            batch_data, batch_labels = data_gen.get_batch()
            if batch_data is None:
                break

            feed_dict = {
                train_data_ph: batch_data,
                train_label_ph: batch_labels,
                is_training_ph: True
            }

            # ========= 动态双向制动对抗训练控制流 =========
            if d_loss_val < 0.01 or g_loss_gan_val > 2.0:
                # 情况 A：D 过强（D_loss 太低）或 G 被压死（G_loss_gan 太高），冻结 D，只更新 G
                g_loss_val, d_loss_val, g_loss_gan_val, _ = sess.run(
                    [g_loss, d_loss, g_loss_gan, g_train_op],
                    feed_dict=feed_dict
                )
            elif d_loss_val > 0.4 or g_loss_gan_val < 0.1:
                # 情况 B：D 太弱被骗（D_loss 太高）或 G 完美欺骗了 D（G_loss_gan 太低），D 多更新一次
                sess.run(d_train_op, feed_dict=feed_dict)
                g_loss_val, d_loss_val, g_loss_gan_val, _, _ = sess.run(
                    [g_loss, d_loss, g_loss_gan, g_train_op, d_train_op],
                    feed_dict=feed_dict
                )
            else:
                # 情况 C：健康对抗状态，标准 1:1 同步更新
                g_loss_val, d_loss_val, g_loss_gan_val, _, _ = sess.run(
                    [g_loss, d_loss, g_loss_gan, g_train_op, d_train_op],
                    feed_dict=feed_dict
                )

            if global_step % args.summary_pred_every == 0:
                g_loss_sum_val, d_loss_sum_val, g_l1_sum_val, g_cos_sum_val, g_tv_sum_val = sess.run(
                    [g_loss_sum, d_loss_sum, g_loss_l1_sum, g_loss_cos_sum, g_loss_tv_sum],
                    feed_dict=feed_dict
                )
                summary_writer.add_summary(g_loss_sum_val, global_step)
                summary_writer.add_summary(d_loss_sum_val, global_step)
                summary_writer.add_summary(g_l1_sum_val, global_step)
                summary_writer.add_summary(g_cos_sum_val, global_step)
                summary_writer.add_summary(g_tv_sum_val, global_step)
            
            if global_step % args.write_pred_every == 0:
                # 可视化时关闭 Dropout，获取干净输出
                gen_val = sess.run(gen_output, feed_dict={
                    train_data_ph: batch_data,
                    is_training_ph: False
                })
                save_prediction(
                    batch_data[0], 
                    batch_labels[0], 
                    gen_val[0], 
                    args.out_dir, 
                    global_step
                )
                save_visual_comparison(
                    batch_data[0], batch_labels[0], gen_val[0],
                    args.out_dir, global_step
                )
            
            if global_step % args.save_pred_every == 0:
                save_checkpoint(saver, sess, global_step)
            
            if global_step % 200 == 0:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f'{current_time} Epoch [{epoch+1}/{args.epoch}] Step [{global_step}] '
                      f'G_loss: {g_loss_val:.4f} D_loss: {d_loss_val:.4f}')
                sys.stdout.flush()
    
    save_checkpoint(saver, sess, global_step)
    print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} [INFO] 训练完成!')
    
    sys.stdout = original_stdout
    log_file.close()
    
    sess.close()


if __name__ == "__main__":
    train()
