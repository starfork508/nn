 import tensorflow as tf
import numpy as np 
import os
import cv2


'''
	- [Tied biases] if you use one bias per convolutional filter/kernel
	- [Untied biases] if you use one bias per kernel and output location.
'''

def mobile_det(image, variable_):
	# stage 1 
	# Conv / s2 [3 × 3 × 3 × 32]
	# input: 300 x 300 x 3
	gamma_scaled_1 = tf.Variable(tf.ones([32]))
	beta_offset_1 = tf.Variable(tf.zeros([32]))
	# -------------
	conv_1 = tf.nn.conv2d(image, variable_["w_1"], [1,2,2,1], padding='SAME', name='conv_1')
	bn_1, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_1, gamma_scaled_1, beta_offset_1)
	relu_1 = tf.nn.relu(bn_1 + variable_["b_1"], name='relu_1')
	# print("relu_1: ", relu_1)

	# stage 2  
	# Conv dw / s1 [3 × 3 × 32 dw]
	# input: 150 x 150 x 32
	gamma_scaled_dw_1 = tf.Variable(tf.ones([32]))
	beta_offset_dw_1 = tf.Variable(tf.constant(0.1, shape=[32]))
	# -------------
	conv_dw_1 = tf.nn.depthwise_conv2d(relu_1, variable_["w_dw_1"], [1,1,1,1], padding='SAME', name='conv_dw_1')
	bn_dw_1, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_dw_1, gamma_scaled_dw_1, beta_offset_dw_1)
	relu_dw_1 = tf.nn.relu(bn_dw_1 + variable_["b_dw_1"], name='relu_dw_1')
	# print("relu_dw_1: ", relu_dw_1)


	# stage 3
	# Conv / s1 [1 × 1 × 32 × 64]
	# input: 150 x 150 x 32
	gamma_scaled_2 = tf.Variable(tf.ones([64]))
	beta_offset_2 = tf.Variable(tf.constant(0.1, shape=[64]))
	# --------------
	conv_2 = tf.nn.conv2d(relu_dw_1, variable_["w_2"], [1,1,1,1], padding='SAME', name='conv_2')
	bn_2, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_2, gamma_scaled_2, beta_offset_2)
	relu_2 = tf.nn.relu(bn_2 + variable_["b_2"], name='relu_2')
	# print("relu_2: ", relu_2)

	# stage 4
	# Conv dw / s2 [3 × 3 × 64 dw]
	# input: 150 x 150 x 64
	gamma_scaled_dw_2 = tf.Variable(tf.ones([64]))
	beta_offset_dw_2 = tf.Variable(tf.constant(0.1, shape=[64]))
	# ---------------
	conv_dw_2 = tf.nn.depthwise_conv2d(relu_2, variable_["w_dw_2"], [1,2,2,1], padding='SAME', name='conv_dw_2')
	bn_dw_2, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_dw_2, gamma_scaled_dw_2, beta_offset_dw_2)
	relu_dw_2 = tf.nn.relu(bn_dw_2 + variable_["b_dw_2"], name='relu_dw_2')
	# print('relu_dw_2: ', relu_dw_2)

	# stage 5
	# Conv / s1 [1 × 1 × 64 × 128]
	# input: 75 x 75 x 64
	gamma_scaled_3 = tf.Variable(tf.ones([128]))
	beta_offset_3 = tf.Variable(tf.constant(0.1, shape=[128]))
	# ----------------
	conv_3 = tf.nn.conv2d(relu_dw_2, variable_["w_3"], [1,1,1,1], padding='SAME', name='conv_3')
	bn_3 , batch_mean, batch_var = tf.nn.fused_batch_norm(conv_3, gamma_scaled_3, beta_offset_3)
	relu_3 = tf.nn.relu(bn_3 + variable_["b_3"], name='relu_3')
	# print('relu_3: ', relu_3)

	# stage 6
	# Conv dw / s1 [3 × 3 × 128 dw]
	# input: 75 x 75 x 128
	gamma_scaled_dw_3 = tf.Variable(tf.ones([128]))
	beta_offset_dw_3 = tf.Variable(tf.constant(0.1, shape=[128]))
	# -------------------
	conv_dw_3 = tf.nn.depthwise_conv2d(relu_3, variable_["w_dw_3"], [1,1,1,1], padding='SAME', name='conv_dw_3')
	bn_dw_3, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_dw_3, gamma_scaled_dw_3, beta_offset_dw_3)
	relu_dw_3 = tf.nn.relu(bn_dw_3 + variable_["b_3"], name='relu_dw_3')
	# print('relu_dw_3: ', relu_dw_3)

	# stage 7
	# Conv / s1 [1 × 1 × 128 × 128]
	# input: 75 x 75 x 128
	gamma_scaled_4 = tf.Variable(tf.ones([128]))
	beta_offset_4 = tf.Variable(tf.constant(0.1, shape=[128]))
	# -------------------
	conv_4 = tf.nn.conv2d(relu_dw_3, variable_["w_4"], [1,1,1,1], padding='SAME', name='conv_4')
	bn_4, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_4, gamma_scaled_4, beta_offset_4)
	relu_4 = tf.nn.relu(bn_4 + variable_["b_4"], name='relu_4')
	# print('relu_4: ', relu_4)

	# stage 8
	# Conv dw / s2 [3 × 3 × 128 dw]
	# input: 75 x 75 x 128
	gamma_scaled_dw_4 = tf.Variable(tf.ones([128]))
	beta_offset_dw_4 = tf.Variable(tf.constant(0.1, shape=[128]))
	# --------------------
	conv_dw_4 = tf.nn.depthwise_conv2d(relu_4, variable_["w_dw_4"], [1,2,2,1], padding='SAME', name='conv_dw_4')
	bn_dw_4, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_dw_4, gamma_scaled_dw_4, beta_offset_dw_4)
	relu_dw_4 = tf.nn.relu(bn_dw_4 + variable_["b_dw_4"], name='relu_dw_4')
	# print('relu_dw_4: ', relu_dw_4)

	# stage 9
	# Conv / s1 [1 × 1 × 128 × 256]
	# input: 38 x 38 x 128
	gamma_scaled_5 = tf.Variable(tf.ones([256]))
	beta_offset_5 = tf.Variable(tf.constant(0.1, shape=[256]))
	# ---------------------
	conv_5 = tf.nn.conv2d(relu_dw_4, variable_["w_5"], [1,1,1,1], padding='SAME')
	bn_5, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_5, gamma_scaled_5, beta_offset_5)
	relu_5 = tf.nn.relu(bn_5 + variable_["b_5"], name='relu_5')
	# print('relu_5: ', relu_5)


	# stage 10 
	# Conv dw / s1 [3 × 3 × 256 dw]
	# input: 38 x 38 x 256
	gamma_scaled_dw_5 = tf.Variable(tf.ones([256]))
	beta_offset_dw_5 = tf.Variable(tf.constant(0.1, shape=[256]))
	# -------------------
	conv_dw_5 = tf.nn.depthwise_conv2d(relu_5, variable_["w_dw_5"], [1,1,1,1], padding='SAME', name='conv_dw_5')
	bn_dw_5, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_dw_5, gamma_scaled_dw_5, beta_offset_dw_5)
	relu_dw_5 = tf.nn.relu(bn_dw_5 + variable_["b_dw_5"], name='relu_dw_5')
	# print('relu_dw_5: ', relu_dw_5)

	# stage 11
	# Conv / s1 [1 × 1 × 256 × 512]
	# input: 38 x 38 x 256
	gamma_scaled_6 = tf.Variable(tf.ones([512]))
	beta_offset_6 = tf.Variable(tf.constant(0.1, shape=[512]))
	# -------------------
	conv_6 = tf.nn.conv2d(relu_dw_5, variable_["w_6"], [1,1,1,1], padding='SAME', name='conv_6')
	bn_6, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_6, gamma_scaled_6, beta_offset_6)
	relu_6 = tf.nn.relu(bn_6 + variable_["b_6"], name='relu_6')
	# print('relu_6: ', relu_6)

	# stage 12
	# Conv dw / s1 [3 × 3 × 512 dw]
	# input: 38 x 38 x 512
	gamma_scaled_dw_6 = tf.Variable(tf.ones([512]))
	beta_offset_dw_6 = tf.Variable(tf.constant(0.1, shape=[512]))
	# --------------------
	conv_dw_6 = tf.nn.depthwise_conv2d(relu_6, variable_["w_dw_6"], [1,1,1,1], padding='SAME', name='conv_dw_6')
	bn_dw_6, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_dw_6, gamma_scaled_6, beta_offset_6)
	relu_dw_6 = tf.nn.relu(bn_dw_6 + variable_["b_dw_6"], name='relu_dw_6')
	# print("relu_dw_6: ", relu_dw_6)

	# stage 13
	# Conv / s1 [1 × 1 × 512 × 512]
	# input: 38 x 38 x 256
	gamma_scaled_7 = tf.Variable(tf.ones([512]))
	beta_offset_7 = tf.Variable(tf.constant(0.1, shape=[512]))
	# ---------------
	conv_7 = tf.nn.conv2d(relu_dw_6, variable_["w_7"], [1,1,1,1], padding='SAME', name='conv_7')
	bn_7, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_7, gamma_scaled_7, beta_offset_7)
	relu_7 = tf.nn.relu(bn_7 + variable_["b_7"], name='relu_7')
	# print('relu_7: ', relu_7)

	# stage 14
	# ....................................
	#	  Conv dw / s1 [3 × 3 × 512 dw]  |  -> input: 38 x 38 x 512
	# 5 ×							   	 | 
	#	  Conv / s1 [1 × 1 × 512 × 512]	 |  -> input: 38 x 38 x 512
	# ....................................
	#
	gamma_scaled_dw_7 = tf.Variable(tf.ones([512]))
	beta_offset_dw_7 = tf.Variable(tf.constant(0.1, shape=[512]))
	#
	gamma_scaled_8 = tf.Variable(tf.ones([512]))
	beta_offset_8 = tf.Variable(tf.constant(0.1, shape=[512]))
	#
	conv_temp = relu_7
	for s in range(5):
		conv_dw_7 = tf.nn.depthwise_conv2d(conv_temp, variable_["w_dw_7"], [1,1,1,1], padding='SAME', name='conv_dw_7')
		bn_dw_7, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_dw_7, gamma_scaled_dw_7, beta_offset_dw_7)
		relu_dw_7 = tf.nn.relu(bn_dw_7 + variable_["b_dw_7"], name='relu_dw_7')

		conv_8 = tf.nn.conv2d(relu_dw_7, variable_["w_8"], [1,1,1,1], padding='SAME', name='conv_8')
		bn_8, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_8, gamma_scaled_8, beta_offset_8)
		relu_8 = tf.nn.relu(bn_8 + variable_["b_8"], name='relu_8')

		conv_temp = relu_8
	relu_8 = conv_temp
	# print('relu_8: ', relu_8)

	# stage 15
	# Conv / s2 [3 × 3 × 512 x 1024]
	# input: 38 x 38 x 512
	gamma_scaled_9 = tf.Variable(tf.ones([1024]))
	beta_offset_9 = tf.Variable(tf.constant(0.1, shape=[1024]))
	# -------------------
	conv_9 = tf.nn.conv2d(relu_8, variable_["w_9"], [1,2,2,1], padding='SAME', name='conv_9')
	bn_9, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_9, gamma_scaled_9, beta_offset_9)
	relu_9 = tf.nn.relu(bn_9 + variable_["b_9"], name='relu_9')
	# print('relu_9: ', relu_9)

	# stage 16
	# Conv / s1 [1 × 1 × 1024 × 1024]
	# input: 19 x 19 x 1024
	gamma_scaled_10 = tf.Variable(tf.ones([1024]))
	beta_offset_10 = tf.Variable(tf.constant(0.1, shape=[1024]))
	# ------------------
	conv_10 = tf.nn.conv2d(relu_9, variable_["w_10"], [1,1,1,1], padding='SAME', name='conv_10')
	bn_10, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_10, gamma_scaled_10, beta_offset_10)
	relu_10 = tf.nn.relu(bn_10 + variable_["b_10"], name='relu_10')
	# print('relu_10: ', relu_10)

	# stage 17
	# Conv / s1 [3 × 3 × 1024 x 256]
	# input: 19 x 19 x 1024
	gamma_scaled_11 = tf.Variable(tf.ones([256]))
	beta_offset_11 = tf.Variable(tf.constant(0.1, shape=[256]))
	# ------------------
	conv_11 = tf.nn.conv2d(relu_10, variable_["w_11"], [1,1,1,1], padding='SAME', name='conv_11')
	bn_11, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_11, gamma_scaled_11, beta_offset_11)
	relu_11 = tf.nn.relu(bn_11 + variable_["b_11"], name='relu_11')
	# print('relu_11: ', relu_11)

	# stage 18
	# Conv / s2 [3 x 3 x 256 x 512]
	# input: 19 x 19 x 256
	gamma_scaled_12 = tf.Variable(tf.ones([512]))
	beta_offset_12 = tf.Variable(tf.constant(0.1, shape=[512]))
	# -----------------
	conv_12 = tf.nn.conv2d(relu_11, variable_["w_12"], [1,2,2,1], padding='SAME', name='conv_12')
	bn_12, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_12, gamma_scaled_12, beta_offset_12)
	relu_12 = tf.nn.relu(bn_12 + variable_["b_12"], name='relu_12')
	# print('relu_12: ', relu_12)

	# stage 19
	# Conv / s1 [1 x 1 x 512 x 128]
	# input: 10 x 10 x 512
	gamma_scaled_13 = tf.Variable(tf.ones([128]))
	beta_offset_13 = tf.Variable(tf.constant(0.1, shape=[128]))
	# -----------------
	conv_13 = tf.nn.conv2d(relu_12, variable_["w_13"], [1,1,1,1], padding='SAME', name='conv_13')
	bn_13, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_13, gamma_scaled_13, beta_offset_13)
	relu_13 = tf.nn.relu(bn_13 + variable_["b_13"], name='relu_13')
	# print('relu_13: ', relu_13)

	# stage 20
	# Conv / s2 [3 x 3 x 128 x 256]
	# input: 10 x 10 x 128
	gamma_scaled_14 = tf.Variable(tf.ones([256]))
	beta_offset_14 = tf.Variable(tf.constant(0.1, shape=[256]))
	# -----------------
	conv_14 = tf.nn.conv2d(relu_13, variable_["w_14"], [1,2,2,1], padding='SAME', name='conv_14')
	bn_14, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_14, gamma_scaled_14, beta_offset_14)
	relu_14 = tf.nn.relu(bn_14 + variable_["b_14"], name='relu_13')
	# print('relu_14: ', relu_14)

	# stage 21
	# Conv / s1 [1 x 1 x 256 x 128]
	# input: 5 x 5 x 256
	gamma_scaled_15 = tf.Variable(tf.ones([128]))
	beta_offset_15 = tf.Variable(tf.constant(0.1, shape=[128]))
	# -----------------
	conv_15 = tf.nn.conv2d(relu_14, variable_["w_15"], [1,1,1,1], padding='SAME', name='conv_15')
	bn_15, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_15, gamma_scaled_15, beta_offset_15)
	relu_15 = tf.nn.relu(bn_15 + variable_["b_15"], name='relu_15')
	# print('relu_15: ', relu_15)

	# stage 22
	# Conv / s2 [3 x 3 x 128 x 256]
	# input: 5 x 5 x 128
	gamma_scaled_16 = tf.Variable(tf.ones([256]))
	beta_offset_16 = tf.Variable(tf.constant(0.1, shape=[256]))
	# -----------------
	conv_16 = tf.nn.conv2d(relu_15, variable_["w_16"], [1,2,2,1], padding='SAME', name='conv_16')
	bn_16, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_16, gamma_scaled_16, beta_offset_16)
	relu_16 = tf.nn.relu(bn_16 + variable_["b_16"], name='relu_16')
	# print('relu_16: ', relu_16)

	# stage 23
	# Conv / s1 [1 x 1 x 256 x 128]
	# input: 3 x 3 x 256
	gamma_scaled_17 = tf.Variable(tf.ones([128]))
	beta_offset_17 = tf.Variable(tf.constant(0.1, shape=[128]))
	# -----------------
	conv_17 = tf.nn.conv2d(relu_16, variable_["w_17"], [1,1,1,1], padding='SAME', name='conv_17')
	bn_17, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_17, gamma_scaled_17, beta_offset_17)
	relu_17 = tf.nn.relu(bn_17 + variable_["b_17"], name='relu_17')
	# print('relu_17: ', relu_17)

	# stage 24
	# Conv / s1 [3 x 3 x 128 x 256]
	# input: 3 x 3 x 128
	gamma_scaled_18 = tf.Variable(tf.ones([256]))
	beta_offset_18 = tf.Variable(tf.constant(0.1, shape=[256]))
	# -----------------
	conv_18 = tf.nn.conv2d(relu_17, variable_["w_18"], [1,1,1,1], padding='VALID', name='conv_18')
	bn_18, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_18, gamma_scaled_18, beta_offset_18)
	relu_18 = tf.nn.relu(bn_18 + variable_["b_18"], name='relu_18')
	# print('relu_18: ', relu_18)

	# stage 25
	# Conv / s1 [1 x 1 x 256 x 128]
	# input: 1 x 1 x 256
	gamma_scaled_19 = tf.Variable(tf.ones([128]))
	beta_offset_19 = tf.Variable(tf.constant(0.1, shape=[128]))
	# -----------------
	conv_19 = tf.nn.conv2d(relu_18, variable_["w_19"], [1,1,1,1], padding='VALID', name='conv_19')
	bn_19, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_19, gamma_scaled_19, beta_offset_19)
	relu_19 = tf.nn.relu(bn_19 + variable_["b_19"], name='relu_18')
	# print('relu_19: ', relu_19)

	# stage 26
	# Conv / s1 [3 x 3 x 128 x 256]
	# input: 1 x 1 x 128
	gamma_scaled_20 = tf.Variable(tf.ones([256]))
	beta_offset_20 = tf.Variable(tf.constant(0.1, shape=[256]))
	# -----------------
	conv_20 = tf.nn.conv2d(relu_19, variable_["w_20"], [1,1,1,1], padding='SAME', name='conv_20')
	bn_20, batch_mean, batch_var = tf.nn.fused_batch_norm(conv_20, gamma_scaled_20, beta_offset_20)
	relu_20 = tf.nn.relu(bn_20 + variable_["b_20"], name='relu_20')
	# print('relu_20: ', relu_20)

	# [38 x 38 x 512]
	# [19 x 19 x 1024]
	# [10 x 10 x 512]
	# [5 x 5 x 256]
	# [3 x 3 x 256]
	# [1 x 1 x 256]
	return [relu_8, relu_10, relu_12, relu_14, relu_16, relu_20]

def variable_mobiledet():
	w_1 = tf.Variable(tf.truncated_normal([3,3,3,32], stddev=0.1), name='w_1')
	b_1 = tf.Variable(tf.constant(1.0, shape=[32]), name='b_1')

	w_dw_1 = tf.Variable(tf.truncated_normal([3,3,32,1], stddev=0.1), name='w_dw_1')
	b_dw_1 = tf.Variable(tf.constant(1.0, shape=[32]), name='b_dw_1')

	w_2 = tf.Variable(tf.truncated_normal([1,1,32,64], stddev=0.1), name='w_2')
	b_2 = tf.Variable(tf.constant(1.0, shape=[64]), name='b_2')

	w_dw_2 = tf.Variable(tf.truncated_normal([3,3,64,1], stddev=0.1), name='w_dw_2')
	b_dw_2 = tf.Variable(tf.constant(1.0, shape=[64]), name='b_dw_2')

	w_3 = tf.Variable(tf.truncated_normal([1,1,64,128], stddev=0.1), name='w_3')
	b_3 = tf.Variable(tf.constant(1.0, shape=[128]), name='b_3')

	w_dw_3 = tf.Variable(tf.truncated_normal([3,3,128,1], stddev=0.1), name='w_dw_3')
	b_dw_3 = tf.Variable(tf.constant(1.0, shape=[128]), name='b_dw_3')

	w_4 = tf.Variable(tf.truncated_normal([1,1,128,128], stddev=0.1), name='w_4')
	b_4 = tf.Variable(tf.constant(1.0, shape=[128]), name='b_4')

	w_dw_4 = tf.Variable(tf.truncated_normal([3,3,128,1], stddev=0.1), name='w_dw_4')
	b_dw_4 = tf.Variable(tf.constant(1.0, shape=[128]), name='b_dw_4')

	w_5 = tf.Variable(tf.truncated_normal([1,1,128, 256], stddev=0.1), name='w_5')
	b_5 = tf.Variable(tf.constant(1.0, shape=[256]), name='b_5')

	w_dw_5 = tf.Variable(tf.truncated_normal([3,3,256,1], stddev=0.1), name='w_dw_5')
	b_dw_5 = tf.Variable(tf.constant(1.0, shape=[256]), name='b_dw_5')

	w_6 = tf.Variable(tf.truncated_normal([1,1,256,512], stddev=0.1), name='w_6')
	b_6 = tf.Variable(tf.constant(1.0, shape=[512]), name='b_6')

	w_dw_6 = tf.Variable(tf.truncated_normal([3,3,512,1], stddev=0.1), name='w_dw_6')
	b_dw_6 = tf.Variable(tf.constant(1.0, shape=[512]), name='b_dw_6')

	w_7 = tf.Variable(tf.truncated_normal([1,1,512,512], stddev=0.1), name='w_7')
	b_7 = tf.Variable(tf.constant(1.0, shape=[512]), name='b_7')

	w_dw_7 = tf.Variable(tf.truncated_normal([3,3,512,1], stddev=0.1), name='w_dw_7')
	b_dw_7 = tf.Variable(tf.constant(1.0, shape=[512]), name='b_dw_7')

	w_8 = tf.Variable(tf.truncated_normal([1,1,512,512], stddev=0.1), name='w_8')
	b_8 = tf.Variable(tf.constant(1.0, shape=[512]), name='b_8')

	w_9 = tf.Variable(tf.truncated_normal([3,3,512,1024], stddev=0.1), name='w_9')
	b_9 = tf.Variable(tf.constant(1.0, shape=[1024]), name='b_9')

	w_10 = tf.Variable(tf.truncated_normal([1,1,1024,1024], stddev=0.1), name='w_10')
	b_10 = tf.Variable(tf.constant(1.0, shape=[1024]), name='b_10')

	w_11 = tf.Variable(tf.truncated_normal([3,3,1024,256], stddev=0.1), name='w_11')
	b_11 = tf.Variable(tf.constant(1.0, shape=[256]), name='b_11')

	w_12 = tf.Variable(tf.truncated_normal([3,3,256,512], stddev=0.1), name='w_12')
	b_12 = tf.Variable(tf.constant(1.0, shape=[512]), name='b_12')

	w_13 = tf.Variable(tf.truncated_normal([1,1,512,128], stddev=0.1), name='w_13')
	b_13 = tf.Variable(tf.constant(1.0, shape=[128]), name='b_13')

	w_14 = tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.1), name='w_14')
	b_14 = tf.Variable(tf.constant(1.0, shape=[256]), name='b_14')

	w_15 = tf.Variable(tf.truncated_normal([1,1,256,128], stddev=0.1), name='w_15')
	b_15 = tf.Variable(tf.constant(1.0, shape=[128]), name='b_15')

	w_16 = tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.1), name='w_16')
	b_16 = tf.Variable(tf.constant(1.0, shape=[256]), name='b_16')

	w_17 = tf.Variable(tf.truncated_normal([1,1,256,128], stddev=0.1), name='w_17')
	b_17 = tf.Variable(tf.constant(1.0, shape=[128]), name='b_17')

	w_18 = tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.1), name='w_18')
	b_18 = tf.Variable(tf.constant(1.0, shape=[256]), name='b_18')

	w_19 = tf.Variable(tf.truncated_normal([1,1,256,128], stddev=0.1), name='w_19')
	b_19 = tf.Variable(tf.constant(1.0, shape=[128]), name='b_19')

	w_20 = tf.Variable(tf.truncated_normal([3,3,128,256], stddev=0.1), name='w_20')
	b_20 = tf.Variable(tf.constant(1.0, shape=[256]), name='b_20')

	variable_mobiledet_ = {"w_1": w_1, "w_2": w_2, "w_3": w_3, "w_4": w_4, "w_5": w_5, "w_6": w_6, "w_7": w_7, "w_8": w_8, "w_9": w_9,
						"w_10": w_10, "w_11": w_11, "w_12": w_12, "w_13": w_13, "w_14": w_14, "w_15": w_15, "w_16": w_16, "w_17": w_17,
						"w_18": w_18, "w_19": w_19, "w_20": w_20,
						"w_dw_1": w_dw_1, "w_dw_2": w_dw_2, "w_dw_3": w_dw_3, "w_dw_4": w_dw_4, "w_dw_5": w_dw_5, "w_dw_6": w_dw_6, "w_dw_7": w_dw_7,
						"b_1": b_1, "b_2": b_2, "b_3": b_3, "b_4": b_4, "b_5": b_5, "b_6": b_6, "b_7": b_7, "b_8": b_8, "b_9": b_9,
						"b_10": b_10, "b_11": b_11, "b_12": b_12, "b_13": b_13, "b_14": b_14, "b_15": b_15, "b_16": b_16, "b_17": b_17,
						"b_18": b_18, "b_19": b_19, "b_20": b_20,
						"b_dw_1": b_dw_1, "b_dw_2": b_dw_2, "b_dw_3": b_dw_3, "b_dw_4": b_dw_4, "b_dw_5": b_dw_5, "b_dw_6": b_dw_6, "b_dw_7": b_dw_7}
	return variable_mobiledet_						

def variable_pred(n_class_, feature_, dict_ar_step):
	n_class_ += 1 # for back-ground
	ar_per_layer = len(dict_ar_step[feature_.name]["ar"]) + 1
	channel_ = int(feature_.name.split('x')[-1].split('_')[0])

	# classifikasi
	conf_w_f_ = tf.Variable(tf.truncated_normal([3,3,channel_, ar_per_layer * n_class_], stddev=0.1))
	conf_b_f_ = tf.Variable(tf.constant(1.0, shape=[ar_per_layer * n_class_]))

	# Regresi
	loc_w_f_ = tf.Variable(tf.truncated_normal([3,3,channel_, ar_per_layer * 4], stddev=0.1))

	variable_pred_ = {"conf_w_f_": conf_w_f_, "conf_b_f_": conf_b_f_, "loc_w_f_": loc_w_f_}
	return variable_pred_

def class_loc_and_dfb(feature_, batch_size_, n_class_, img_size_, n_feature_map, dict_ar_step, variable_pred_):
	n_class_ += 1 # for back-ground
	ar_per_layer = len(dict_ar_step[feature_.name]["ar"]) + 1
	channel_ = int(feature_.name.split('x')[-1].split('_')[0])
	feature_size = int(feature_.name.split('x')[1])

	# classifikasi
	pc_feature_ = tf.nn.conv2d(feature_, variable_pred_["conf_w_f_"], [1,1,1,1], padding='SAME', name='class_'+feature_.name.split(':')[0])
	conf_softmax_ = tf.nn.softmax(pc_feature_ + variable_pred_["conf_b_f_"], name='conf_softmax_'+feature_.name.split(':')[0].split('_')[1])

	# regresi
	loc_class_feature_ = tf.nn.conv2d(feature_, variable_pred_["loc_w_f_"], [1,1,1,1], padding='SAME', name='class_'+feature_.name.split(':')[0])

	# default box generator
	scale_min = 0.2
	scale_max = 0.9
	scale_k = scale_min + (scale_max - scale_min)*(dict_ar_step[feature_.name]["urut"] - 1) / (N_FEATURE_MAP - 1)
	scale_kp1 = scale_min + (scale_max - scale_min)*(dict_ar_step[feature_.name]["urut"]) / (N_FEATURE_MAP - 1)
	scale_k_prime = (scale_k * scale_kp1) ** 0.5

	start_center = 0.5 * dict_ar_step[feature_.name]["step"]
	finish_center = img_size_ - start_center
	center_x = np.linspace(start_center, finish_center, feature_size)
	center_y = np.linspace(start_center, finish_center, feature_size)

	# center_x_full = center_x + np.zeros((center_x.shape[0], center_x.shape[0]))
	# center_y_full = np.expand_dims(center_y, axis=-1) + np.zeros(center_y.shape[0], center_y.shape[0])

	center_x_grid, center_y_grid = np.meshgrid(center_x, center_y)

	center_x_grid_exp = np.expand_dims(center_x_grid, axis=-1)
	center_y_grid_exp = np.expand_dims(center_y_grid, axis=-1)

	width_cell_list = np.append(scale_k * np.sqrt(dict_ar_step[feature_.name]["ar"]), scale_k_prime)
	height_cell_list = np.append(scale_k / np.sqrt(dict_ar_step[feature_.name]["ar"]), scale_k_prime)

	width_cell_list *= dict_ar_step[feature_.name]["step"]
	height_cell_list *= dict_ar_step[feature_.name]["step"]

	width_height = np.array(list(set(zip(width_cell_list, height_cell_list))))

	dfb_ = np.zeros([feature_size, feature_size, ar_per_layer, 4]) # 4 -> cx, cy, width, height

	dfb_[:,:,:,0] = np.tile(center_x_grid_exp, (1,1,ar_per_layer))
	dfb_[:,:,:,1] = np.tile(center_y_grid_exp, (1,1,ar_per_layer))
	dfb_[:,:,:,2] = width_height[:,0]
	dfb_[:,:,:,3] = width_height[:,1]

	# untuk semua batch size
	dfbes = np.expand_dims(dfb_, axis=0)
	dfbes_tf = tf.tile(tf.constant(dfbes, dtype=tf.float32), (batch_size_, 1,1,1,1))
	
	return conf_softmax_, loc_class_feature_, dfbes_tf


def tf_cvt_center2ujung(tf_center_bbox_): # xmin, ymin, xmax, ymax
	ujung_bbox_0 = tf_center_bbox_[:,:,0] - tf.scalar_mul(0.5, tf_center_bbox_[:,:,2])
	ujung_bbox_1 = tf_center_bbox_[:,:,1] - tf.scalar_mul(0.5, tf_center_bbox_[:,:,3])
	ujung_bbox_2 = tf_center_bbox_[:,:,0] + tf.scalar_mul(0.5, tf_center_bbox_[:,:,2])
	ujung_bbox_3 = tf_center_bbox_[:,:,1] + tf.scalar_mul(0.5, tf_center_bbox_[:,:,3])

	e0 = tf.expand_dims(ujung_bbox_0, axis=-1)
	e1 = tf.expand_dims(ujung_bbox_1, axis=-1)
	e2 = tf.expand_dims(ujung_bbox_2, axis=-1)
	e3 = tf.expand_dims(ujung_bbox_3, axis=-1)
	tf_concat_ujung = tf.concat([e0,e1,e2,e3], axis=-1)
	tf_concat_ujung_clipped = tf.clip_by_value(t=tf_concat_ujung, clip_value_min=0.0, clip_value_max=np.inf)
	return tf_concat_ujung_clipped

def tf_compute_iou_fast(tf_full_gt, tf_full_df):
	mask_1 = tf.logical_or(tf.greater(tf_full_gt[:, :, 0], tf_full_df[:, :, 2]), tf.less(tf_full_gt[:, :, 2], tf_full_df[:, :, 0]))
	mask_2 = tf.logical_or(mask_1, tf.less(tf_full_gt[:, :, 3], tf_full_df[:, :, 1]))
	mask_3 = tf.logical_or(mask_2, tf.greater(tf_full_gt[:, :, 1], tf_full_df[:, :, 3]))
	mask_fix = tf.logical_not(mask_3)

	int_mask_fix = tf.cast(mask_fix, tf.float32)

	old_shape = tf_full_gt.get_shape().as_list()
	new_shape = list(np.array([old_shape[0], np.prod(old_shape[1:-1])], np.int32))

	in_one_mask_fix = tf.reshape(int_mask_fix, new_shape)

	tlx = tf.maximum(tf_full_gt[:,:,0], tf_full_df[:,:,0])
	tly = tf.maximum(tf_full_gt[:,:,1], tf_full_df[:,:,1])
	brx = tf.minimum(tf_full_gt[:,:,2], tf_full_df[:,:,2])
	bry = tf.minimum(tf_full_gt[:,:,3], tf_full_df[:,:,3])
	area_overlap = tf.cast(tf.abs(brx - tlx) * tf.abs(bry - tly), tf.float32)
	area_overlap = tf.multiply(area_overlap, in_one_mask_fix)

	area_gt = tf.cast(tf.abs(tf_full_gt[:, :, 2] - tf_full_gt[:, :, 0]) * tf.abs(tf_full_gt[:, :, 3] - tf_full_gt[:, :, 1]), tf.float32)
	area_df = tf.cast(tf.abs(tf_full_df[:, :, 2] - tf_full_df[:, :, 0]) * tf.abs(tf_full_df[:, :, 3] - tf_full_df[:, :, 1]), tf.float32)
	union_ = area_gt + area_df - area_overlap
	iou_ = area_overlap / union_
	return iou_

def tf_filter_by_iou_fast(tf_dfbes, gt_boxes, thresh_iou):
	'''
		code_pos_loc_ -> terhadap ground-truth
		code_pos_conf_ -> terhadap label
		conf_neg_ -> sisa
	'''
	batch_size_ = tf_dfbes.get_shape().as_list()[0]
	nbox_default = tf_dfbes.get_shape().as_list()[1]

	code_pos_loc_full = list(list() for _ in range(batch_size_))
	code_pos_conf_full = list(list() for __ in range(batch_size_))
	code_neg_full = list(list() for ___ in range(batch_size_))
	# iou_full = list()

	for b in range(batch_size_):
		# gt_boxes[b] -> (n_gt_box, 4)
		exp_gt_b = tf.expand_dims(gt_boxes[b], axis=-1) # (n_gt_box, 4, 1)
		tf_tile_item_gt = tf.tile(exp_gt_b, (1,1,nbox_default)) # (n_gt_box, 4, nbox_default)
		transposed_gt_item = tf.transpose(tf_tile_item_gt, perm=(0,2,1)) # (n_gt_box, nbox_default, 4)

		exp_df_ = tf.expand_dims(tf_dfbes[b], axis=0)
		tf_tile_df = tf.tile(exp_df_, (len(gt_boxes[b]),1,1))

		iou_ = tf_compute_iou_fast(transposed_gt_item, tf_tile_df)
		
		cond_chosen = (iou_ >= thresh_iou)
		code_pos_loc_ = tf.where(cond_chosen)

		goto_where_pos = tf.reduce_any(cond_chosen, axis=0)
		code_pos_conf_  = tf.where(goto_where_pos)
		code_neg_  = tf.where(tf.logical_not(goto_where_pos))
		
		code_pos_loc_full[b].append(code_pos_loc_)
		code_pos_conf_full[b].append(code_pos_conf_)
		code_neg_full[b].append(code_neg_)

	return code_pos_loc_full, code_pos_conf_full, code_neg_full

def tf_hnm(batch_code_neg_conf_, pred_conf_full, n_neg_required):
	batch_size_ = len(batch_code_neg_conf_)
	
	rawsort_conf_ = list()
	chosen_index = list(list() for _ in range(batch_size_))
	if not n_neg_required:
		return None

	for b in range(batch_size_):
		for code_in_b in batch_code_neg_conf_[b]:
			rawsort_conf_.append(pred_conf_full[b][code_in_b][0])
		tensor_raw = tf.constant(rawsort_conf_)
		sorted_index = tf.nn.top_k(tensor_raw, k=n_neg_required[b], sorted=True)
		chosen_index[b].append(sorted_index[1])
		rawsort_conf_ = list()
	return chosen_index

def operate_two_box(one_gt_box, one_df_box):
	prep_gt_ = np.zeros_like(one_gt_box)
	prep_gt_[0] = float(abs(one_gt_box[0] - one_df_box[0]))/one_df_box[2]
	prep_gt_[1] = float(abs(one_gt_box[1] - one_df_box[1]))/one_df_box[3]
	prep_gt_[2] = np.log10(max(one_gt_box[2], 1e-15)/max(one_df_box[2], 1e-15))
	prep_gt_[3] = np.log10(max(one_gt_box[3], 1e-15)/max(one_df_box[3], 1e-15))
	return prep_gt_

def cvt_ujung2center(ujung_bbox_, tipe):
	center_bbox_ = np.zeros_like(ujung_bbox_)
	if tipe == '3d':
		center_bbox_[:,:,0] = 0.5*(ujung_bbox_[:,:,0] + ujung_bbox_[:,:,2])
		center_bbox_[:,:,1] = 0.5*(ujung_bbox_[:,:,1] + ujung_bbox_[:,:,3])
		center_bbox_[:,:,2] = abs(ujung_bbox_[:,:,2] - ujung_bbox_[:,:,0])
		center_bbox_[:,:,3] = abs(ujung_bbox_[:,:,3] - ujung_bbox_[:,:,1])
	elif tipe == '2d':
		center_bbox_[:,0] = 0.5*(ujung_bbox_[:,0] + ujung_bbox_[:,2])
		center_bbox_[:,1] = 0.5*(ujung_bbox_[:,1] + ujung_bbox_[:,3])
		center_bbox_[:,2] = abs(ujung_bbox_[:,2] - ujung_bbox_[:,0])
		center_bbox_[:,3] = abs(ujung_bbox_[:,3] - ujung_bbox_[:,1])
	return center_bbox_

def prepocess_box(gt_box_, df_box_, pred_loc_box_, code_pos_loc_):
	batch_size_ = len(code_pos_loc_)
	n_class_ = len(code_pos_loc_[0])

	center_format_gt = list(cvt_ujung2center(np.array(gt_box_[b]), '2d') for b in range(batch_size_))

	center_format_df = cvt_ujung2center(df_box_, '3d')
	center_pred_loc = cvt_ujung2center(pred_loc_box_,'3d')

	prep_gt = list()
	prep_pred = list()

	for b in range(batch_size_):
		for c in range(n_class_):
			for tpl in code_pos_loc_[b][c]:
				goto_prep_gt = operate_two_box(center_format_gt[b][tpl[0]], center_format_df[b][tpl[1]])
				prep_gt.append(goto_prep_gt)
				prep_pred.append(pred_loc_box_[b][tpl[1]])
	arr_prep_gt = np.array(prep_gt)
	arr_prep_pred = np.array(prep_pred)
	return arr_prep_gt, arr_prep_pred

def tf_compute_conf_loss(code_pos_conf_, code_neg_conf_, tf_pc_softmax_):
	n_class_ = len(code_pos_conf_[0])
	batch_size_ = len(code_pos_conf_)
	Lconf_value = tf.zeros((1), tf.float32)
	min_tensor = tf.constant(1e-15, tf.float32)
	
	for b in range(batch_size_):
		for c in range(n_class_):
			for lp in range(len(code_pos_conf_[b][c])):
				Lconf_value -= tf.log(tf.maximum(tf_pc_softmax_[b][code_pos_conf_[b][c][lp]][c], min_tensor))/tf.log(10.0)
		for ln in range(len(code_neg_conf_[b])):
			Lconf_value -= tf.log(tf.maximum(tf_pc_softmax_[b][code_neg_conf_[b][ln]][0], min_tensor))/tf.log(10.0)
	return Lconf_value

def compute_loc_loss(prep_gt_, prep_pred_):
	'''
	 Smooth L1 loss
	 https://blog.csdn.net/u014365862/article/details/79924201
	'''

	value_condition_1 = 0.5 * (prep_pred_ - prep_gt_)**2
	value_condition_2 = np.abs(prep_pred_ - prep_gt_)

	smooth_l1_loss = np.where(value_condition_2 < 1.0, value_condition_1, value_condition_2 - 0.5)
	Lloc_value = np.sum(smooth_l1_loss)
	return Lloc_value

def compute_all_loss(loc_loss_, conf_loss_, alpha_, n_pos_box_):
	if not n_pos_box_:
		return 0.0
	total_loss = (conf_loss_ + alpha_*loc_loss_)/n_pos_box_
	return total_loss

def one_hot_encoder(n, n_class_):
	nol_ = np.zeros(n_class_+1)
	nol_[n] = 1
	return nol_


# ---------------------------------------
# config parameter
# ---------------------------------------
batch_size = 3
img_size = img_width = img_height = 300
img_depth = 3
n_class = 3
n_classP1 = n_class + 1
N_FEATURE_MAP = 6 # bawaan SSD
alpha__ = 1.0
learning_rate_ = 0.03
max_step_ = 10000

# filter by iou
threshold_iou = 0.25

# hard negative mining
n_neg_per_pos = 3.0/1.0
#----------------------------------------



alamat_img_train_face_24 = "images_mobilenet/train_face_24.jpg"
alamat_img_train_hand_199 = "images_mobilenet/train_hand_199.jpg"
alamat_img_train_bottle_81 = "images_mobilenet/train_bottle_81.jpg"

img_train_face_24 = cv2.imread(alamat_img_train_face_24)
img_train_face_24 = cv2.resize(img_train_face_24, (img_width,img_height))

img_train_hand_199 = cv2.imread(alamat_img_train_hand_199)
img_train_hand_199 = cv2.resize(img_train_hand_199, (img_width,img_height))

img_train_bottle_81 = cv2.imread(alamat_img_train_bottle_81)
img_train_bottle_81 = cv2.resize(img_train_bottle_81, (img_width,img_height))

np_data_img = np.array([img_train_face_24, img_train_hand_199, img_train_bottle_81])

### Get bounding box
alamat_xml_train_face_24 = "images_mobilenet/train_face_24.xml"
alamat_xml_train_hand_199 = "images_mobilenet/train_hand_199.xml"
alamat_xml_train_bottle_81 = "images_mobilenet/train_bottle_81.xml"

import xml.etree.ElementTree as ET 

gt_labels = list(list() for b in range(batch_size))
gt_box = list(list() for b in range(batch_size))


dataset_xml = (alamat_xml_train_face_24, alamat_xml_train_hand_199, alamat_xml_train_bottle_81)

dict_labels_ = {"face": 1, "hand": 2, "bottle": 3}

for i in range(len(dataset_xml)):
	tree_ = ET.parse(dataset_xml[i])
	root = tree_.getroot()

	for member in root.findall('object'):
		xmin = float(member[4][0].text) * float(img_width)/float(root.find('size')[0].text)
		ymin = float(member[4][1].text) * float(img_height)/float(root.find('size')[1].text)
		xmax = float(member[4][2].text) * float(img_width)/float(root.find('size')[0].text)
		ymax = float(member[4][3].text) * float(img_height)/float(root.find('size')[1].text)
		gt_box[i].append([xmin, ymin, xmax, ymax])
		class_ = member[0].text
		ohe_label = one_hot_encoder(dict_labels_[class_], n_class)
		gt_labels[i].append(ohe_label)
# print("gt_box: ", gt_box)



graph = tf.Graph()
with graph.as_default():
	train_data = tf.placeholder(tf.float32, shape=(batch_size, img_width, img_height, img_depth), name='train_data')
	
	variable_model_in_graph = variable_mobiledet()
	feature_mobile_det = mobile_det(train_data, variable_model_in_graph)
	
	f_bx38x38x512_plc = tf.placeholder(tf.float32, shape=(batch_size,38,38,512), name='f_bx38x38x512_plc')
	f_bx19x19x1024_plc = tf.placeholder(tf.float32, shape=(batch_size,19,19,1024), name='f_bx19x19x1024_plc')
	f_bx10x10x512_plc = tf.placeholder(tf.float32, shape=(batch_size,10,10,512), name='f_bx10x10x512_plc')
	f_bx5x5x256_plc = tf.placeholder(tf.float32, shape=(batch_size,5,5,256), name='f_bx5x5x256_plc')
	f_bx3x3x256_plc = tf.placeholder(tf.float32, shape=(batch_size,3,3,256), name='f_bx3x3x256_plc')
	f_bx1x1x256_plc = tf.placeholder(tf.float32, shape=(batch_size,1,1,256), name='f_bx1x1x256_plc')

	dict_ar_step = {f_bx38x38x512_plc.name: {"ar": [1.0, 2.0, 0.5], "step": 8, "urut": 1},
				   f_bx19x19x1024_plc.name: {"ar": [1.0, 2.0, 0.5, 3.0, 1.0/3.0], "step": 16, "urut": 2},
				   f_bx10x10x512_plc.name: {"ar": [1.0, 2.0, 0.5, 3.0, 1.0/3.0], "step": 32, "urut": 3},
				   f_bx5x5x256_plc.name: {"ar": [1.0, 2.0, 0.5, 3.0, 1.0/3.0], "step": 64, "urut": 4},
				   f_bx3x3x256_plc.name: {"ar": [1.0, 2.0, 0.5], "step": 100, "urut": 5},
				   f_bx1x1x256_plc.name: {"ar": [1.0, 2.0, 0.5], "step": 300, "urut": 6}}

	variable_pred_in_graph = variable_pred
	v_bx38x38x512 = class_loc_and_dfb(f_bx38x38x512_plc, batch_size,
										n_class_= n_class,
										img_size_= img_size,
										n_feature_map= N_FEATURE_MAP,
										dict_ar_step= dict_ar_step,
										variable_pred_=variable_pred_in_graph(n_class, f_bx38x38x512_plc, dict_ar_step))

	v_bx19x19x1024 = class_loc_and_dfb(f_bx19x19x1024_plc, batch_size,
										n_class_= n_class,
										img_size_= img_size, 
										n_feature_map= N_FEATURE_MAP, 
										dict_ar_step= dict_ar_step, 
										variable_pred_=variable_pred_in_graph(n_class, f_bx19x19x1024_plc, dict_ar_step))

	v_bx10x10x512 = class_loc_and_dfb(f_bx10x10x512_plc, batch_size, 
										n_class_= n_class, 
										img_size_= img_size, 
										n_feature_map= N_FEATURE_MAP, 
										dict_ar_step= dict_ar_step, 
										variable_pred_=variable_pred_in_graph(n_class, f_bx10x10x512_plc, dict_ar_step))

	v_bx5x5x256 = class_loc_and_dfb(f_bx5x5x256_plc, batch_size, 
										n_class_= n_class, 
										img_size_= img_size, 
										n_feature_map= N_FEATURE_MAP, 
										dict_ar_step= dict_ar_step, 
										variable_pred_=variable_pred_in_graph(n_class, f_bx5x5x256_plc, dict_ar_step))

	v_bx3x3x256 = class_loc_and_dfb(f_bx3x3x256_plc, batch_size,
									n_class_= n_class, 
									img_size_= img_size, 
									n_feature_map= N_FEATURE_MAP, 
									dict_ar_step= dict_ar_step, 
									variable_pred_=variable_pred_in_graph(n_class, f_bx3x3x256_plc, dict_ar_step))

	v_bx1x1x256 = class_loc_and_dfb(f_bx1x1x256_plc, batch_size, 
									n_class_= n_class, 
									img_size_= img_size, 
									n_feature_map= N_FEATURE_MAP, 
									dict_ar_step= dict_ar_step, 
									variable_pred_=variable_pred_in_graph(n_class, f_bx1x1x256_plc, dict_ar_step))

	ndfb_layer = [5776, 2166, 600, 150, 36, 4]
	dfb_v_bx5776x4 = tf.reshape(v_bx38x38x512[2], [batch_size, ndfb_layer[0], 4])
	dfb_v_bx2166x4 = tf.reshape(v_bx19x19x1024[2], [batch_size, ndfb_layer[1], 4])
	dfb_v_bx600x4 = tf.reshape(v_bx10x10x512[2], [batch_size, ndfb_layer[2], 4])
	dfb_v_bx150x4 = tf.reshape(v_bx5x5x256[2], [batch_size, ndfb_layer[3], 4])
	dfb_v_bx36x4 = tf.reshape(v_bx3x3x256[2], [batch_size, ndfb_layer[4], 4])
	dfb_v_bx4x4 = tf.reshape(v_bx1x1x256[2], [batch_size, ndfb_layer[5], 4])

	npb_layer = [5776, 2166, 600, 150, 36, 4]
	pb_v_bx5776x4 = tf.reshape(v_bx38x38x512[1], [batch_size, npb_layer[0], 4])
	pb_v_bx2166x4 = tf.reshape(v_bx19x19x1024[1], [batch_size, npb_layer[1], 4])
	pb_v_bx600x4 = tf.reshape(v_bx10x10x512[1], [batch_size, npb_layer[2], 4])
	pb_v_bx150x4 = tf.reshape(v_bx5x5x256[1], [batch_size, npb_layer[3], 4])
	pb_v_bx36x4 = tf.reshape(v_bx3x3x256[1], [batch_size, npb_layer[4], 4])
	pb_v_bx4x4 = tf.reshape(v_bx1x1x256[1], [batch_size, npb_layer[5], 4])

	npc_layer = [5776, 2166, 600, 150, 36, 4]
	pc_v_bx5776xnclassP1 = tf.reshape(v_bx38x38x512[0], [batch_size, npc_layer[0], n_classP1])
	pc_v_bx2166xnclassP1 = tf.reshape(v_bx19x19x1024[0], [batch_size, npc_layer[1], n_classP1])
	pc_v_bx600xnclassP1 = tf.reshape(v_bx10x10x512[0], [batch_size, npc_layer[2], n_classP1])
	pc_v_bx150xnclassP1 = tf.reshape(v_bx5x5x256[0], [batch_size, npc_layer[3], n_classP1])
	pc_v_bx36xnclassP1 = tf.reshape(v_bx3x3x256[0], [batch_size, npc_layer[4], n_classP1])
	pc_v_bx4xnclassP1 = tf.reshape(v_bx1x1x256[0], [batch_size, npc_layer[5], n_classP1])

	print(pc_v_bx4xnclassP1.get_shape().as_list())
	dfb_v_bx8732x4 = tf.concat([dfb_v_bx5776x4, dfb_v_bx2166x4, dfb_v_bx600x4, dfb_v_bx150x4, dfb_v_bx36x4, dfb_v_bx4x4], 1)
	pb_v_bx8732x4 = tf.concat([pb_v_bx5776x4, pb_v_bx2166x4, pb_v_bx600x4, pb_v_bx150x4, pb_v_bx36x4, pb_v_bx4x4],1)
	pc_v_bx8732xnclassP1 = tf.concat([pc_v_bx5776xnclassP1, pc_v_bx2166xnclassP1, pc_v_bx600xnclassP1, pc_v_bx150xnclassP1, pc_v_bx36xnclassP1, pc_v_bx4xnclassP1], 1)

	ujung_v_bx8732x4 = tf_cvt_center2ujung(dfb_v_bx8732x4)

	v_code_pos_locANDconf_v_code_neg_ = tf_filter_by_iou_fast(ujung_v_bx8732x4,	gt_box,	threshold_iou)

with tf.Session(graph=graph) as sess:
	tf.global_variables_initializer().run()

	for step_ in range(1):#range(max_step_):
		print("step_", step_)
		feed_feature_mobile_det_ = {train_data: np_data_img}
		feature_mobile_det_ = sess.run([feature_mobile_det], feed_dict= feed_feature_mobile_det_)

		f_bx38x38x512_fd = {f_bx38x38x512_plc: feature_mobile_det_[0][0]}
		f_bx19x19x1024_fd = {f_bx19x19x1024_plc: feature_mobile_det_[0][1]}
		f_bx10x10x512_fd = {f_bx10x10x512_plc: feature_mobile_det_[0][2]}
		f_bx5x5x256_fd = {f_bx5x5x256_plc: feature_mobile_det_[0][3]}
		f_bx3x3x256_fd = {f_bx3x3x256_plc: feature_mobile_det_[0][4]}
		f_bx1x1x256_fd = {f_bx1x1x256_plc: feature_mobile_det_[0][5]}

		pc_nv_bx38x38x_4xnclassP1, pb_nv_bx38x38x16, dfb_nv_bx38x38x4x4 = sess.run(v_bx38x38x512, feed_dict= f_bx38x38x512_fd)
		pc_nv_bx19x19x_6xnclassP1, pb_nv_bx19x19x24, dfb_nv_bx19x19x6x4 = sess.run(v_bx19x19x1024, feed_dict= f_bx19x19x1024_fd)
		pc_nv_bx10x10x_6xnclassP1, pb_nv_bx10x10x24, dfb_nv_bx10x10x6x4 = sess.run(v_bx10x10x512, feed_dict= f_bx10x10x512_fd)
		pc_nv_bx5x5x_6xnclassP1, pb_nv_bx5x5x24, dfb_nv_bx5x5x6x4 = sess.run(v_bx5x5x256, feed_dict= f_bx5x5x256_fd)
		pc_nv_bx3x3x_4xnclassP1, pb_nv_bx3x3x16, dfb_nv_bx3x3x4x4 = sess.run(v_bx3x3x256, feed_dict= f_bx3x3x256_fd)
		pc_nv_bx1x1x_4xnclassP1, pb_nv_bx1x1x16, dfb_nv_bx1x1x4x4 = sess.run(v_bx1x1x256, feed_dict= f_bx1x1x256_fd)

		fd_v_bx38x38x512 = {v_bx38x38x512: (pc_nv_bx38x38x_4xnclassP1, pb_nv_bx38x38x16, dfb_nv_bx38x38x4x4)}
		fd_v_bx19x19x1024 = {v_bx19x19x1024: (pc_nv_bx19x19x_6xnclassP1, pb_nv_bx19x19x24, dfb_nv_bx19x19x6x4)} 
		fd_v_bx10x10x512 = {v_bx10x10x512: (pc_nv_bx10x10x_6xnclassP1, pb_nv_bx10x10x24, dfb_nv_bx10x10x6x4)}
		fd_v_bx5x5x256 = {v_bx5x5x256: (pc_nv_bx5x5x_6xnclassP1, pb_nv_bx5x5x24, dfb_nv_bx5x5x6x4)}
		fd_v_bx3x3x256 = {v_bx3x3x256: (pc_nv_bx3x3x_4xnclassP1, pb_nv_bx3x3x16, dfb_nv_bx3x3x4x4)}
		fd_v_bx1x1x256 = {v_bx1x1x256: (pc_nv_bx1x1x_4xnclassP1, pb_nv_bx1x1x16, dfb_nv_bx1x1x4x4)}

		dfb_nv_bx5776x4 = sess.run(dfb_v_bx5776x4, feed_dict=fd_v_bx38x38x512)
		dfb_nv_bx2166x4 = sess.run(dfb_v_bx2166x4, feed_dict=fd_v_bx19x19x1024)
		dfb_nv_bx600x4 = sess.run(dfb_v_bx600x4, feed_dict=fd_v_bx10x10x512)
		dfb_nv_bx150x4 = sess.run(dfb_v_bx150x4, feed_dict=fd_v_bx5x5x256)
		dfb_nv_bx36x4 = sess.run(dfb_v_bx36x4, feed_dict=fd_v_bx3x3x256)
		dfb_nv_bx4x4 = sess.run(dfb_v_bx4x4, feed_dict=fd_v_bx1x1x256)

		pb_nv_bx5776x4 = sess.run(pb_v_bx5776x4, feed_dict=fd_v_bx38x38x512)
		pb_nv_bx2166x4 = sess.run(pb_v_bx2166x4, feed_dict=fd_v_bx19x19x1024)
		pb_nv_bx600x4 = sess.run(pb_v_bx600x4, feed_dict=fd_v_bx10x10x512)
		pb_nv_bx150x4 = sess.run(pb_v_bx150x4, feed_dict=fd_v_bx5x5x256)
		pb_nv_bx36x4 = sess.run(pb_v_bx36x4, feed_dict=fd_v_bx3x3x256)
		pb_nv_bx4x4 = sess.run(pb_v_bx4x4, feed_dict=fd_v_bx1x1x256)

		pc_nv_bx5776xnclassP1 = sess.run(pc_v_bx5776xnclassP1, feed_dict=fd_v_bx38x38x512)
		pc_nv_bx2166xnclassP1 = sess.run(pc_v_bx2166xnclassP1, feed_dict=fd_v_bx19x19x1024)
		pc_nv_bx600xnclassP1 = sess.run(pc_v_bx600xnclassP1, feed_dict=fd_v_bx10x10x512)
		pc_nv_bx150xnclassP1 = sess.run(pc_v_bx150xnclassP1, feed_dict=fd_v_bx5x5x256)
		pc_nv_bx36xnclassP1 = sess.run(pc_v_bx36xnclassP1, feed_dict=fd_v_bx3x3x256)
		pc_nv_bx4xnclassP1 = sess.run(pc_v_bx4xnclassP1, feed_dict=fd_v_bx1x1x256)

		fd_concat_dfb = {dfb_v_bx5776x4: dfb_nv_bx5776x4, 
						 dfb_v_bx5776x4: dfb_nv_bx5776x4, 
						 dfb_v_bx600x4: dfb_nv_bx600x4, 
						 dfb_v_bx150x4: dfb_nv_bx150x4, 
						 dfb_v_bx36x4: dfb_nv_bx36x4, 
						 dfb_v_bx4x4: dfb_nv_bx4x4}

		fd_concat_pb = {pb_v_bx5776x4: pb_nv_bx5776x4, 
						pb_v_bx2166x4: pb_nv_bx2166x4, 
						pb_v_bx600x4: pb_nv_bx600x4, 
						pb_v_bx150x4: pb_nv_bx150x4, 
						pb_v_bx36x4: pb_nv_bx36x4, 
						pb_v_bx4x4: pb_nv_bx4x4}

		fd_concat_pc = {pc_v_bx5776xnclassP1: pc_nv_bx5776xnclassP1, 
						pc_v_bx2166xnclassP1: pc_nv_bx2166xnclassP1, 
						pc_v_bx600xnclassP1: pc_nv_bx600xnclassP1, 
						pc_v_bx150xnclassP1: pc_nv_bx150xnclassP1, 
						pc_v_bx36xnclassP1: pc_nv_bx36xnclassP1, 
						pc_v_bx4xnclassP1: pc_nv_bx4xnclassP1}

		dfb_nv_bx8732x4 = sess.run(dfb_v_bx8732x4, feed_dict= fd_concat_dfb)
		pb_nv_bx8732x4 = sess.run(pb_v_bx8732x4, feed_dict= fd_concat_pb)
		pc_nv_bx8732xnclassP1 = sess.run(pc_v_bx8732xnclassP1, feed_dict= fd_concat_pc)

		# print("dfb_nv_bx8732x4: ", dfb_nv_bx8732x4)
		fd_ctr2ujung= {dfb_v_bx8732x4: dfb_nv_bx8732x4}

		ujung_nv_bx8732x4 = sess.run(ujung_v_bx8732x4, feed_dict=fd_ctr2ujung)
		# print("ujung_nv_bx8732x4: ", ujung_nv_bx8732x4)

		fd_ujung_ = {ujung_v_bx8732x4: ujung_nv_bx8732x4}
		
		code_pos_loc_full, code_pos_conf_full, code_neg_full = sess.run(v_code_pos_locANDconf_v_code_neg_, feed_dict=fd_ujung_)

		for b in range(batch_size):
			for cd_ in code_pos_conf_full[b][0]:
				ind_ = cd_[0]
				rect_ = ujung_nv_bx8732x4[b][ind_]
				cv2.rectangle(np_data_img[b], (rect_[0], rect_[1]), (rect_[2], rect_[3]), (255,0,0))
			for mm in gt_box[b]:
				cv2.rectangle(np_data_img[b], (int(mm[0]), int(mm[1])), (int(mm[2]), int(mm[3])), (0,0,255))

			cv2.imshow(str(b), np_data_img[b])
		cv2.waitKey(0)

		# n_neg_minimum_ = list()
		# for code_pos_ in code_pos_conf_full:
		# 	n = 0
		# 	for c in code_pos_:
		# 		n += len(c)
		# 	n_neg_minimum_.append(n_neg_per_pos * n)

		# neg_after_hmn_ = tf_hnm(ses_code_neg_conf_, pc_nv_bx8732xnclassP1, n_neg_minimum_)

		# ses_conf_loss_ = tf_compute_conf_loss(ses_code_pos_conf_, neg_after_hmn_, pc_nv_bx8732xnclassP1)
		# print("conf_loss: ", ses_conf_loss_)

		# arr_prep_gt_, arr_prep_pred_ = prepocess_box(gt_box, dfb_nv_bx8732x4, pb_nv_bx8732x4, ses_code_pos_loc_)
		# ses_loc_loss_ = compute_loc_loss(arr_prep_gt_, arr_prep_pred_)
		# print("loc_loss: ", ses_loc_loss_)

		# n_pos_loc_ = arr_prep_gt_.shape[0]
		# ses_all_loss_ = compute_all_loss(ses_loc_loss_, ses_conf_loss_, alpha__, n_pos_loc_)
		# print("all_loss_: ", ses_all_loss_, "\n\n")