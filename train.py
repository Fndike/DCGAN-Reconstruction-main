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

import numpy as np
import os
import argparse
import tensorflow as tf
import glob
import random

from net import Generator, Discriminator

parser = argparse.ArgumentParser(description='DCGAN 3D Geological Model Reconstruction Training')

parser.add_argument("--train_data_dir", default='./dataset/processed/', help="Directory containing training data files")
parser.add_argument("--snapshot_dir", default='./model_save', help='Path to save model checkpoints')
parser.add_argument("--out_dir", default='./train_out', help='Path to save training outputs')
parser.add_argument("--image_size", type=int, default=64, help="Image size (width and height)")
parser.add_argument("--image_size_z", type=int, default=64, help="Image size (depth)")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
parser.add_argument("--epoch", type=int, default=200, help="Number of training epochs")
parser.add_argument("--base_lr_g", type=float, default=0.0002, help="Learning rate for generator")
parser.add_argument("--base_lr_d", type=float, default=0.0002, help="Learning rate for discriminator")
parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
parser.add_argument("--random_seed", type=int, default=1234, help="Random seed")
parser.add_argument("--save_pred_every", type=int, default=1000, help="Save model every N steps")
parser.add_argument("--summary_pred_every", type=int, default=100, help="Save summary every N steps")
parser.add_argument("--write_pred_every", type=int, default=500, help="Write prediction every N steps")
parser.add_argument("--lambda_l1", type=float, default=100.0, help="L1 loss weight")
parser.add_argument("--lambda_gan", type=float, default=1.0, help="GAN loss weight")

args = parser.parse_args()
EPS = 1e-12


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
    
    if len(data.shape) == 4:
        data = np.expand_dims(data, axis=-1)
        data = np.concatenate([data] * 3, axis=-1)
    if len(labels.shape) == 4:
        labels = np.expand_dims(labels, axis=-1)
        labels = np.concatenate([labels] * 3, axis=-1)
    
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
    print(f'[INFO] 模型已保存: step {step}')


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


def train():
    """主训练函数"""
    tf.set_random_seed(args.random_seed)
    create_directories()
    
    data_gen = DataGenerator(args.train_data_dir, args.batch_size, prefix="train")
    total_samples = len(data_gen)
    steps_per_epoch = total_samples // args.batch_size
    
    print(f"[INFO] 总训练样本数: {total_samples}")
    print(f"[INFO] 每epoch步数: {steps_per_epoch}")
    
    train_data_ph = tf.placeholder(
        tf.float32, 
        shape=[None, args.image_size, args.image_size, args.image_size_z, 3],
        name='train_data'
    )
    train_label_ph = tf.placeholder(
        tf.float32, 
        shape=[None, args.image_size, args.image_size, args.image_size_z, 3],
        name='train_label'
    )
    
    print("[INFO] 构建生成器...")
    gen_output = Generator(image_3D=train_data_ph, gf_dim=64, reuse=False, name='generator')
    
    print("[INFO] 构建判别器...")
    dis_real = Discriminator(train_data_ph, train_label_ph, df_dim=64, reuse=False, name='discriminator')
    dis_fake = Discriminator(train_data_ph, gen_output, df_dim=64, reuse=True, name='discriminator')
    
    print("[INFO] 计算损失函数...")
    g_loss_gan = tf.reduce_mean(-tf.log(dis_fake + EPS))
    g_loss_l1 = tf.reduce_mean(tf.abs(gen_output - train_label_ph))
    g_loss = args.lambda_gan * g_loss_gan + args.lambda_l1 * g_loss_l1
    
    d_loss_real = tf.reduce_mean(-tf.log(dis_real + EPS))
    d_loss_fake = tf.reduce_mean(-tf.log(1 - dis_fake + EPS))
    d_loss = d_loss_real + d_loss_fake
    
    g_loss_sum = tf.summary.scalar('generator_loss', g_loss)
    d_loss_sum = tf.summary.scalar('discriminator_loss', d_loss)
    g_loss_l1_sum = tf.summary.scalar('generator_l1_loss', g_loss_l1)
    
    gen_vars = [v for v in tf.trainable_variables() if 'generator' in v.name]
    dis_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name]
    
    print(f"[INFO] 生成器参数数量: {sum([np.prod(v.shape) for v in gen_vars])}")
    print(f"[INFO] 判别器参数数量: {sum([np.prod(v.shape) for v in dis_vars])}")
    
    g_optimizer = tf.train.AdamOptimizer(args.base_lr_g, beta1=args.beta1)
    d_optimizer = tf.train.AdamOptimizer(args.base_lr_d, beta1=args.beta1)
    
    g_train_op = g_optimizer.minimize(g_loss, var_list=gen_vars)
    d_train_op = d_optimizer.minimize(d_loss, var_list=dis_vars)
    train_op = tf.group(d_train_op, g_train_op)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())
    
    print(f"[INFO] 开始训练...")
    
    global_step = 0
    
    for epoch in range(args.epoch):
        data_gen.reset()
        
        for step in range(steps_per_epoch):
            global_step += 1
            
            batch_data, batch_labels = data_gen.get_batch()
            if batch_data is None:
                break
            
            feed_dict = {
                train_data_ph: batch_data,
                train_label_ph: batch_labels
            }
            
            g_loss_val, d_loss_val, _ = sess.run(
                [g_loss, d_loss, train_op], 
                feed_dict=feed_dict
            )
            
            if global_step % args.summary_pred_every == 0:
                g_loss_sum_val, d_loss_sum_val, g_l1_sum_val = sess.run(
                    [g_loss_sum, d_loss_sum, g_loss_l1_sum],
                    feed_dict=feed_dict
                )
                summary_writer.add_summary(g_loss_sum_val, global_step)
                summary_writer.add_summary(d_loss_sum_val, global_step)
                summary_writer.add_summary(g_l1_sum_val, global_step)
            
            if global_step % args.write_pred_every == 0:
                gen_val = sess.run(gen_output, feed_dict=feed_dict)
                save_prediction(
                    batch_data[0], 
                    batch_labels[0], 
                    gen_val[0], 
                    args.out_dir, 
                    global_step
                )
            
            if global_step % args.save_pred_every == 0:
                save_checkpoint(saver, sess, global_step)
            
            if global_step % 50 == 0:
                print(f'Epoch [{epoch+1}/{args.epoch}] Step [{global_step}] '
                      f'G_loss: {g_loss_val:.4f} D_loss: {d_loss_val:.4f}')
    
    save_checkpoint(saver, sess, global_step)
    print("[INFO] 训练完成!")
    
    sess.close()


if __name__ == "__main__":
    train()
