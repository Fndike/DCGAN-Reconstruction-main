import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pan
import math
import matplotlib as plt


'''
构造可训练参数
'''
def make_var(name,shape,trainable = True):
    return tf.get_variable(name,shape,trainable=trainable)

'''
定义卷积层（第三维度的数据可以类比于视频处理的关键帧）
'''
def conv3D(input_,output_dim,kernel_size,stride,padding = "SAME",name = "conv3d",biased = False):
    input_dim = input_.get_shape()[-1]#读出输入层的维度
    with tf.variable_scope(name):
        kernal = make_var(name = 'weights',shape = [kernel_size,kernel_size,kernel_size,input_dim,output_dim]) #定义卷积块
        output = tf.nn.conv3d(input_,
                              kernal,
                              [1,stride,stride,stride,1],
                              padding=padding)      #定义卷积过程
        if biased:
            biases = make_var(name = 'biases',shape=[output_dim])  #偏差
            output = tf.nn.bias_add(output,biases) #将偏差加到value上面

    return output

'''
定义反卷积层
'''
def deconv3D(input_, output_dim, kernel_size, stride, padding="SAME", name="deconv3d"):
    input_dim = int(input_.get_shape()[-1])
    input_shape = tf.shape(input_)

    with tf.variable_scope(name):
        kernel = make_var(
            name='weights',
            shape=[kernel_size, kernel_size, kernel_size, output_dim, input_dim]
        )
        output_shape = tf.stack([
            input_shape[0],
            input_shape[1] * stride,
            input_shape[2] * stride,
            input_shape[3] * stride,
            output_dim
        ])

        output = tf.nn.conv3d_transpose(
            input_,
            kernel,
            output_shape,
            [1, stride, stride, stride, 1],
            padding=padding
        )

    return output

'''
定义归一化函数BN层(维度问题需要检查)
'''
def batch_norm(input_,name = 'batch_norm'):
    with tf.variable_scope(name):
        input_dim = input_.get_shape()[-1]
        print("控制点，检查是否为后两项的计算失误",name,input_dim)
        scale = tf.get_variable("scale",
                                [input_dim],
                                initializer=tf.random_normal_initializer(1.0,0.02,dtype=tf.float32))
        offset = tf.get_variable("offset",
                                 [input_dim],
                                 initializer=tf.constant_initializer(0.0))
        print("offset,scale = ",offset,scale)
        mean,variance = tf.nn.moments(input_,axes=[1,2,3],keep_dims=True)
        print("mean,variance = ",mean,variance)
        epsilom = 1e-5
        inv = tf.rsqrt(variance + epsilom)
        normalized = (input_ - mean) * inv
        output = scale * normalized + offset
        test = scale * normalized
        print("各种变量的维度","scale",scale.shape,"offset",offset.shape,"normalized",normalized.shape,"相乘的维度",test.shape,"最终output的维度",output.shape)
        return output

'''
激活层：利用leakyrelu函数激活避免梯度爆炸和消失
'''
def lrelu(x,leak = 0.2,name = 'relu'):
    return tf.maximum(x,leak * x)  #relu函数本质上就是一个取大值的函数


def spectral_norm(weight, name="spectral_norm", n_iters=1):
    """对卷积核做谱归一化，使用幂迭代近似谱范数。
    weight shape: [k,k,k,in_ch,out_ch]
    返回归一化后的 weight / sigma。
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w_shape = weight.shape.as_list()
        w_mat = tf.reshape(weight, [-1, w_shape[-1]])  # [k*k*k*in_ch, out_ch]
        u_var = tf.get_variable(
            "u",
            shape=[1, w_shape[-1]],
            initializer=tf.truncated_normal_initializer(),
            trainable=False
        )
        u = u_var
        for _ in range(n_iters):
            v = tf.nn.l2_normalize(tf.matmul(u, tf.transpose(w_mat)), axis=None)
            u = tf.nn.l2_normalize(tf.matmul(v, w_mat), axis=None)
        sigma = tf.matmul(tf.matmul(v, w_mat), tf.transpose(u))[0, 0]
        w_sn = weight / sigma
        return w_sn


def conv3D_sn(input_, output_dim, kernel_size, stride, padding="SAME", name="conv3d_sn", biased=False):
    """带谱归一化的 3D 卷积（无 batch norm）。"""
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name='weights', shape=[kernel_size, kernel_size, kernel_size, input_dim, output_dim])
        kernel_sn = spectral_norm(kernel, name='sn')
        output = tf.nn.conv3d(input_, kernel_sn, [1, stride, stride, stride, 1], padding=padding)
        if biased:
            biases = make_var(name='biases', shape=[output_dim])
            output = tf.nn.bias_add(output, biases)
    return output


'''
进行生成器的输出（128的数据两层卷积的U-Net）
'''
def Generator(image_3D,gf_dim = 64,reuse = False,is_training = None,name = 'generator'):
    input_dim = int(image_3D.get_shape()[-1])

    # 如果外部没有传占位符，默认创建一个默认行为
    # 如果传入的是 Python bool（如 inference 传 False），自动转为 tf.constant
    if isinstance(is_training, bool):
        is_training = tf.constant(is_training, dtype=tf.bool)
    elif is_training is None:
        is_training = tf.placeholder_with_default(True, shape=(), name='is_training_default')

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # 下采样（5层，适配 Z=32 输入）
        # e1: [None, 64, 64, 32, input_dim] → [None, 32, 32, 16, gf_dim]
        e1 = batch_norm(conv3D(input_=image_3D, output_dim=gf_dim, kernel_size=4, stride=2, name='g_conv_e1'),
                        name='g_bn_e1')
        print("e1 shape:", e1.shape)

        # e2: [None, 32, 32, 16, gf_dim] → [None, 16, 16, 8, gf_dim*2]
        e2 = batch_norm(conv3D(input_=lrelu(e1), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_conv_e2'),
                        name='g_bn_e2')
        print("e2 shape:", e2.shape)

        # e3: [None, 16, 16, 8, gf_dim*2] → [None, 8, 8, 4, gf_dim*4]
        e3 = batch_norm(conv3D(input_=lrelu(e2), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_conv_e3'),
                        name='g_bn_e3')
        print("e3 shape:", e3.shape)

        # e4: [None, 8, 8, 4, gf_dim*4] → [None, 4, 4, 2, gf_dim*8]
        e4 = batch_norm(conv3D(input_=lrelu(e3), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_conv_e4'),
                        name='g_bn_e4')
        print("e4 shape:", e4.shape)

        # e5: [None, 4, 4, 2, gf_dim*8] → [None, 2, 2, 1, gf_dim*8]
        e5 = batch_norm(conv3D(input_=lrelu(e4), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_conv_e5'),
                        name='g_bn_e5')
        print("e5 shape:", e5.shape)

        # ======= 上采样阶段的 Dropout 动态控制 =======
        # 动态切换 keep_prob: 训练时为 0.5，推理时为 1.0（不丢弃）
        current_keep_prob = tf.cond(is_training, lambda: 0.5, lambda: 1.0)

        # d2: [None, 2, 2, 1, gf_dim*8] → [None, 4, 4, 2, gf_dim*8], concat e4
        d2 = deconv3D(input_=tf.nn.relu(e5), output_dim=gf_dim * 8, kernel_size=4, stride=2, name='g_deconv_d2')
        d2 = tf.nn.dropout(d2, current_keep_prob)
        d2 = tf.concat([batch_norm(input_=d2, name='g_bn_d2'), e4], 4)
        print("d2 shape:", d2.shape)

        # d3: [None, 4, 4, 2, gf_dim*8*2] → [None, 8, 8, 4, gf_dim*4], concat e3
        d3 = deconv3D(input_=tf.nn.relu(d2), output_dim=gf_dim * 4, kernel_size=4, stride=2, name='g_deconv_d3')
        d3 = tf.nn.dropout(d3, current_keep_prob)
        d3 = tf.concat([batch_norm(input_=d3, name='g_bn_d3'), e3], 4)
        print("d3 shape:", d3.shape)

        # d4: [None, 8, 8, 4, gf_dim*4*2] → [None, 16, 16, 8, gf_dim*2], concat e2
        d4 = deconv3D(input_=tf.nn.relu(d3), output_dim=gf_dim * 2, kernel_size=4, stride=2, name='g_deconv_d4')
        d4 = tf.nn.dropout(d4, current_keep_prob)
        d4 = tf.concat([batch_norm(input_=d4, name='g_bn_d4'), e2], 4)
        print("d4 shape:", d4.shape)

        # d5: [None, 16, 16, 8, gf_dim*2*2] → [None, 32, 32, 16, gf_dim], 不拼接 e1，阻断网格特征泄露
        d5 = deconv3D(input_=tf.nn.relu(d4), output_dim=gf_dim, kernel_size=4, stride=2, name='g_deconv_d5')
        d5 = tf.nn.dropout(d5, current_keep_prob)
        d5 = batch_norm(input_=d5, name='g_bn_d5')
        print("d5 shape:", d5.shape)

        # 【修改后】：d_final 拆分为两步
        # 第一步：用 deconv3D 撑开分辨率，并保留足够的特征厚度 (gf_dim // 2)，保证边界锐利度
        d_up = deconv3D(input_=tf.nn.relu(d5), output_dim=gf_dim // 2, kernel_size=4, stride=2, name='g_deconv_d6_up')
        print("d_up shape:", d_up.shape)

        # 第二步：接入 stride=1 的标准卷积作为"熨斗"，抹平反卷积网格噪点，压缩到 1 通道
        d_final = conv3D(input_=lrelu(d_up), output_dim=1, kernel_size=3, stride=1, name='g_smooth_d6')
        print("d_final shape:", d_final.shape)

        return tf.nn.tanh(d_final)



'''
定义discriminator：四层（暂定，不知道3D的生成精准度比2D高多少,层数不能太深避免训练失衡）
'''
#参数之中的image_3D为输入的原数据（train_data），targets是目标数据(train_label/gen_label)
def Discriminator(image_3D,targets,df_dim = 64,reuse = False,name = 'discriminator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        dis_input = tf.concat([image_3D,targets],4)#不是很明白为什么要进行合并，找其他的卷积神经网络看一下
        print("判别器输入的数据为 = ",dis_input.shape)
        #第一层卷积
        print("判别器第一层网络输入前的维度",dis_input)
        dis0 = lrelu(conv3D(input_=dis_input,output_dim=df_dim,kernel_size=4,stride=2,name='dis_con_0'))
        print("判别器第一层网络结束后的维度",dis0)
        #第2层卷积

        dis1 = lrelu(conv3D_sn(input_=dis0,output_dim=df_dim*2,kernel_size=4,stride=2,name='dis_conv_1'))
        print("判别器第二层网络后的维度", dis1)
        #第3层卷积

        dis2 = lrelu(conv3D_sn(input_=dis1,output_dim=df_dim*4,kernel_size=4,stride=2,name='dis_conv_2'))
        print("第三层网络结束后的维度",dis2)
        #第4层卷积

        dis3 = lrelu(conv3D_sn(input_=dis2,output_dim=df_dim*4,kernel_size=4,stride=2,name='dis_conv_3'))
        print("第三层网络结束后的维度",dis3)
        #最终层卷积

        dis_output = conv3D_sn(input_=dis3,output_dim=1,kernel_size=4,stride=1,name='dis_conv_output')
        print("最终层网络结束后的维度",dis_output)
        #经过sigmoid层进行运算，作用为进行二分类运算，输出的结果为分类的结果
        dis_output = tf.sigmoid(dis_output)
        print("最终结果",dis_output)
        return dis_output
