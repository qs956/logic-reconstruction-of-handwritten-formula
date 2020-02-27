#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: https://github.com/qs956

import numpy as np
import pandas as pd
import math
import re
import warnings


#--------------------------------------------------------------------------------------------------#
#不能作为上下标的逻辑
logiclist0=['\\in','=','+','\\times','\\div','\\int','\\rightarrow','\\neq','>','<','\\geq','\\ge',
'\\ldots','\\pm','\\leq',']','}','\\exists',')',',','\\gt',',','\\lt','\\forall','\\in','.','\\dot',
'\\','|','!','\\prime']#up
logiclist1=['\\in','=','+','\\times','\\div','\\int','\\rightarrow','\\neq','>','<','\\geq','\\ge',
'\\ldots','\\pm','\\leq',']','}','\\exists',')',',','-','\\gt',',','\\lt','\\forall','\\in','.',
'\\dot','\\','|','!','\\prime']#down
logiclist2=['\\in','=','+','-','\\times','\\div','\\rightarrow','\\neq','>','<','\\geq','\\ge',
'\\ldots','\\pm','\\leq','{','[','(','\\exists','\\gt',',','\\lt','\\forall','\\in','.','\\dot',
'\\','|','!','\\prime']#主体
not_single=logiclist1 + ['(','[','{']

#集合可以加快查找速度
logiclist0,logiclist1,logiclist2,not_single = set(logiclist0),set(logiclist1),set(logiclist2),set(not_single)

'''
#--------------------------------------------------------------------------------------------------#
以下为图像逻辑识别
'''
#具有类似分数上下关系的符号
frac_like = [r'\sum',r'\lim']

def sort_df(df):
	df.sort_values(by = ['x','y'],inplace = True)
	return df

def common_rebuil(df):
	df = sort_df(df)
	s = ''
	for i in df['name']:
		s += i
	return s

def rec_overlap_s(x1min,x1max,y1min,y1max,x2min,x2max,y2min,y2max):
	#给定两个矩形的坐标,计算两个矩形的重叠面积
	if ((min(x1min,x1max) > max(x2min,x2max)) or (max(x1min,x1max) < min(x2min,x2max))):
		ans1 = 0
	else:
		x = np.sort(np.array([x1min,x1max,x2min,x2max]))
		ans1 = abs(x[2]-x[1])
	if ((min(y1min,y1max) > max(y2min,y2max)) or (max(y1min,y1max) < min(y2min,y2max))):
		ans2 = 0
	else:
		y = np.sort(np.array([y1min,y1max,y2min,y2max]))
		ans2 = abs(y[2]-y[1])
	ans =  ans1*ans2
	return ans

def dist(x1,y1,x2,y2):
	#欧氏距离
	return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def right_where(df1,df2,pic_shape = (1024,1024)):
	#判断在df1右边的df2的具体位置
	#首先根据字符高度差判断主体和客体
	#如果主体合理且客体合理,进入上下标重构的判断程序,否则直接输右边
	if (df1['name'] not in logiclist2):
		#进一步根据客体距离主体的最高点中点以及最低点判断客体位置
		L1 = dist(df1['xmax'],df1['ymax'],df2['x'],df2['y'])
		L2 = dist((df1['xmax']+df1['xmin'])/2,(df1['ymax']+df1['ymin'])/2,df2['x'],df2['y'])
		L3 = dist(df1['xmax'],df1['ymin'],df2['x'],df2['y'])
		#根据竖直距离判断上下标
		if ((L1 < L2) or (L3 < L2)):
		# if (1):
			h1 = dist(0,df1['ymax'],0,df2['y'])
			h2 = dist(0,df1['y'],0,df2['y'])
			h3 = dist(0,df1['ymin'],0,df2['y'])
			if (h1 <= h2):
				if (df2['name'] not in logiclist0):
					#右上角
					s1 = rec_overlap_s(pic_shape[1],df1['xmax'],0,df1['ymax'],df2['xmin'],df2['xmax'],df2['ymin'],df2['ymax'])
					#正上方
					s2 = rec_overlap_s(df1['xmin'],df1['xmax'],0,df1['ymax'],df2['xmin'],df2['xmax'],df2['ymin'],df2['ymax'])
					if (s1 < s2):
						return 8
					else:
						return 7
				else:
					return 6
			elif (h3 <= h2):
				if (df2['name'] not in logiclist1 and df1['name'] not in set([str(i) for i in range(10)])):
					#右下角
					s1 = rec_overlap_s(pic_shape[1],df1['xmax'],df1['ymin'],-pic_shape[0],df2['xmin'],df2['xmax'],df2['ymin'],df2['ymax'])
					#正下方
					s2 = rec_overlap_s(df1['xmin'],df1['xmax'],df1['ymin'],-pic_shape[0],df2['xmin'],df2['xmax'],df2['ymin'],df2['ymax'])
					if (s2 > s1):
						return 4
					else:
						return 5
				else:
					return 6
			else:
				return 6
		else:
			return 6
	else:
		return 6
	return 6

def int_where(df,m):
	if (r'\int' not in df['name']):
		return m
	index = np.where(df['name'] == r'\int')[0]
	# for i in index:
	return m

def frac_where(df1,df2,pic_shape = (1024,1024)):
	assert df1['name'] in ['-',r'\sum',r'\lim'],'未定义规则!'
	# if not (df1['xmin'] <= df2['x'] <= df1['xmax']):
	if ((df1['xmin'] >= df2['xmax']) or (df1['xmax'] <= df2['xmin'])):
		return right_where(df1,df2,pic_shape)
	#判断df2在分数线\求和df1的具体位置
	if (df2['ymin'] >= df1['ymin']):
		return 8
	elif (df2['ymax'] <= df1['ymax']):
		return 4
		#这两种情况才可能是分数线或者是求和
	else:
		return right_where(df1,df2,pic_shape)

def sqrt_where(df1,df2,pic_shape = (1024,1024)):
	#判断df1是否包含df2
	assert df1['name'] == r'\sqrt','非根号无法重构!'
	s = np.zeros((6))
	#正下方
	s[0] = rec_overlap_s(df1['xmin'],df1['xmax'],df1['ymin'],-pic_shape[0],df2['xmin'],df2['xmax'],df2['ymin'],df2['ymax'])
	#右下标
	s[1] = rec_overlap_s(pic_shape[1],df1['xmax'],df1['ymin'],-pic_shape[0],df2['xmin'],df2['xmax'],df2['ymin'],df2['ymax'])
	#右方
	s[2] = rec_overlap_s(pic_shape[1],df1['xmax'],df1['ymin'],df1['ymax'],df2['xmin'],df2['xmax'],df2['ymin'],df2['ymax'])
	#右上标
	s[3] = rec_overlap_s(pic_shape[1],df1['xmax'],0,df1['ymax'],df2['xmin'],df2['xmax'],df2['ymin'],df2['ymax'])
	#正上方
	s[4] = rec_overlap_s(df1['xmin'],df1['xmax'],0,df1['ymax'],df2['xmin'],df2['xmax'],df2['ymin'],df2['ymax'])
	#里面
	s[5] = rec_overlap_s(df1['xmin'],df1['xmax'],df1['ymin'],df1['ymax'],df2['xmin'],df2['xmax'],df2['ymin'],df2['ymax'])
	if (np.argmax(s) == 5):
		return 9
	else :
		return right_where(df1,df2,pic_shape)

def get_edge_label(df1,df2,pic_shape = (1024,1024)):
	#计算2相对于1的位置关系,并返回权重表示
	if (df1['xmin'] >= df2['xmax']):
		return 0
	if (df1['name'] == r'\sqrt'):
		return sqrt_where(df1,df2,pic_shape)
	elif(df1['name'] in frac_like + [r'-']):
		return frac_where(df1,df2)
	else:
		return right_where(df1,df2,pic_shape)

def keep_first(m,b):
	#把序列m中第一个为b的值保留,其余的赋值为0,返回m
	index = np.where(m == b)[0]
	if (len(index) != 0):
		m[index] = 0
		m[index[0]] = b
	return m

def frac_like_expand(df,pic_shape = (1024,1024)):
	#用于对特殊符号的范围扩展以提高识别准确率
	if (df.shape[0] == 0):
		return df
	ans = []
	for i in frac_like:
		index = np.where(df['name'] == i)[0]
		if (len(index) == 0):
			continue
		index = df.index[index]
		for j in index:
			temp = df[np.logical_not((df['ymin'] > df.loc[j,'ymax']) | (df['ymax'] < df.loc[j,'ymin']))]
			temp = temp.copy()
			temp.drop(j,inplace = True)
			left = temp[temp['xmax'] < df.loc[j,'xmin']]['xmax'].values
			left = 0 if len(left) == 0 else np.max(left)
			right = temp[temp['xmin'] > df.loc[j,'xmax']]['xmin'].values
			right = pic_shape[0] if len(right) == 0 else np.min(right)
			ans.append((j,left,right))
	for j in ans:
		pos,left,right = j
		df.loc[pos,'xmin'] = left
		df.loc[pos,'xmax'] = right
	return df

def frac_expand(df,pic_shape = (1024,1024)):
	#分数线的范围扩展以提高识别效率
	if (df.shape[0] == 0):
		return df
	index = np.where(df['name'] == r'-')[0]
	if (len(index) == 0):
			return df
	ans = []
	for i in index:
		#先计算这个分数原来的上下部分
		m = np.zeros(df.shape[0])
		for j in range(df.shape[0]):
			if (i == j):
				m[j] = 0
			else:
				m[j] = get_edge_label(df.loc[df.index[i]],df.loc[df.index[j]],pic_shape)
		up = np.where(m == 8)[0]
		down = np.where(m == 4)[0]
		if (len(up) == 0) and (len(down) == 0):#这时候是减号
			continue
		up_ylow = df.loc[df.index[up],'ymin'].min()
		down_yhigh = df.loc[df.index[down],'ymax'].max()
		if (up_ylow <= down_yhigh):#分数线是斜的,或者是分数线矩形框宽度为0,方法不适合
			continue
		temp = df[np.logical_not((df['ymin'] > up_ylow) | (df['ymax'] < down_yhigh))]
		temp = temp.copy()
		temp.drop(df.index[i],inplace = True)
		left = temp[temp['xmax'] <= df.loc[df.index[i],'xmin']]['xmax'].values
		left = 0 if len(left) == 0 else np.max(left)
		right = temp[temp['xmin'] >= df.loc[df.index[i],'xmax']]['xmin'].values
		right = pic_shape[0] if len(right) == 0 else np.min(right)

		ans.append((i,left,right))
	for j in ans:
		pos,left,right = j
		df.loc[df.index[pos],'xmin'] = left
		df.loc[df.index[pos],'xmax'] = right
	return df

def get_grahp_matrix(df,pic_shape = (1024,1024)):
	#构造图的邻接矩阵
	m = np.zeros((df.shape[0],df.shape[0]))
	df = df.copy()
	if (m.shape[0] == 1):
		return m
	for i in range(df.shape[0]):
		for j in range(df.shape[0]):
			if (i == j):
				m[i,j] = 0
			else:
				m[i,j] = get_edge_label(df.loc[df.index[i]],df.loc[df.index[j]],pic_shape)
	#使用最短欧氏距离匹配右连连接关系
	for i in range(df.shape[0]):
		temp = m[i,:]
		right = np.where(temp == 6)[0]
		d = np.zeros_like(right)
		for j in range(len(right)):
			d[j] = dist(df.loc[df.index[i],'x'],df.loc[df.index[i],'y'],df.loc[df.index[right[j]],'x'],df.loc[df.index[right[j]],'y'])
		if (len(d) != 0):
			min_index = np.argmin(d)
			temp[right] = 0
			temp[right[min_index]] = 6
			m[i,:] = temp
	#保留第一个
	for i in range(df.shape[0]):
		# for j in [4,5,7,8]:
		for j in [5,7]:
			m[i,:] = keep_first(m[i,:],j)
	return m

def get_right_list(m):
	#给定标记的邻接矩阵,返回矩阵中所有的右连列表
	if (m.shape[0] == 1): return [[0]]
	m = (m == 6)
	ans = []
	label = [i for i in range(m.shape[0])]
	while (len(label) != 0 ):
		list1 = []
		head = label[0]
		list1.append(head)
		label.remove(head)
		while (1):
			pos = np.where(m[head,:] == 1)[0]
			if (len(pos) == 0 or (pos[0] not in label)):
				break
			else:
				head = pos[0]
				list1.append(head)
				label.remove(head)
		ans.append(list1)
	return ans

def list_insert(l,obj1,obj2):
	#在列表l对象obj1的后面插入对象列表obj2
	pos = 0
	for i in range(len(l)):
		if (l[i] == obj1):
			pos = i+1
			break
	for i in range(len(obj2)-1,-1,-1):
		l.insert(pos,obj2[i])
	return l

def list_to_nameid(df,l):
	#给定序列把其转换为字符串
	ans = ''
	for i in l:
		if (type(i) == str):
			ans += i
		else:
			ans += str(df['name'].values[i]) + ' '
	return ans

def up_down_complement(word_list,mode):
	#对上下标字符串补全Latex代码
	assert mode in ['up','down'],'不适用于上下标补全!'
	assert type(word_list) == list,'请输入字符ID的列表！'
	if (mode == 'up'):
		temp = [r'^{'] + word_list +[r'}']
	elif (mode == 'down'):
		temp = [r'_{'] + word_list + [r'}']
	return temp

def get_csv_attribute(x_list, y_list):
	x = x_list
	y = y_list
	xmid = (x.max() + x.min()) / 2
	ymid = (y.max() + y.min()) / 2
	ymax = y.max()
	ymin = y.min()
	xmax = x.max()
	xmin = x.min()
	return xmid,ymid,ymax,ymin,xmax,xmin

def get_relevance_attribute(df,relevance):
	x_list = df.loc[relevance,['xmax','xmin','x']].values
	y_list = df.loc[relevance,['ymax','ymin','y']].values
	return list(get_csv_attribute(x_list,y_list))

def frac_rebuil(df,pic_shape = (1024,1024)):
	#分号重构
	frac_may_index = np.where(df['name'] == '-')[0]
	if (len(frac_may_index) == 0):#没有分号可能
		return df
	while (len(frac_may_index)):
		m = get_grahp_matrix(df)
		# right_list = get_right_list(m)
		index = frac_may_index[0]
		if not((8 in m[index,:]) and (4 in m[index,:])):
			frac_may_index = frac_may_index[1:]#这表明是减号
			continue
		up = np.where(m[index,:] == 8)[0]
		down = np.where(m[index,:] == 4)[0]
		up_list = df.index[up]
		down_list = df.index[down]
		up_ans = rebuil(df.loc[up_list,:])
		down_ans = rebuil(df.loc[down_list,:])

		ans = r'\frac{' + up_ans + '}{' + down_ans + '}'
		relevance = [df.index[index]] + list(up_list) + list(down_list)
		df.loc[df.index[index]] = [ans] + get_relevance_attribute(df,relevance)
		df.drop(relevance[1:],inplace = True)
		df = sort_df(df)
		df = df.reset_index(drop = True)
		frac_may_index = np.where(df['name'] == '-')[0]
	df = df.reset_index(drop = True)
	return df

def sqrt_rebuil(df,pic_shape = (1024,1024)):
	#根号重构
	sqrt_may_index = np.where(df['name'] == r'\sqrt')[0]
	if (len(sqrt_may_index) == 0):#没有根号可能
		return df
	while (len(sqrt_may_index)):
		m = get_grahp_matrix(df)
		# right_list = get_right_list(m)
		index = sqrt_may_index[0]

		if 9 not in m[index,:]:
			warnings.warn('根号中无内容,识别过程可能存在错误!', UserWarning)
			df.loc[df.index[index],"name"] = r'\sqrt{}'
		else:
			contain = np.where(m[index,:] == 9)[0]
			relevance = list(df.index[contain])
			ans = r'\sqrt{' + rebuil(df.loc[relevance]) + '}'
			df.loc[df.index[index]] = [ans] + get_relevance_attribute(df,[df.index[index]] + relevance)
			df.drop(relevance,inplace = True)
		df = sort_df(df)
		df = df.reset_index(drop = True)
		sqrt_may_index = np.where(df['name'] == r'\sqrt')[0]
	df = df.reset_index(drop = True)
	return df

def up_down_rebuil(df,pic_shape = (1024,1024)):
	# print(df)
	m = get_grahp_matrix(df)
	# print(m)
	index = get_right_list(m)
	ans_list = index[0]
	for i in range(1,len(index)):
		start = index[i][0]
		pos1 = np.where(m[:,start] == 5)[0]#上标
		pos2 = np.where(m[:,start] == 7)[0]#下标
		# assert (len(pos1) != 0) or (len(pos2) != 0),'未定义规则!'
		if (len(pos1) == 0) and (len(pos2) == 0):
			ans_list += index[i]
		elif (len(pos2) == 0):#上标
			if (len(index[i]) == 1) and (df.loc[df.index[index[i][0]],'name'] in not_single):
				temp = index[i]
			else:
				temp = up_down_complement(index[i] , 'down')
			list_insert(ans_list,pos1[-1],temp)
		elif (len(pos1) == 0):
			if (len(index[i]) == 1) and (df.loc[df.index[index[i][0]],'name'] in not_single):
				temp = index[i]
			else:
				temp = up_down_complement(index[i] , 'up')
			list_insert(ans_list,pos2[-1],temp)
		elif (pos1[-1] > pos2[-1]):
			if (len(index[i]) == 1) and (df.loc[df.index[index[i][0]],'name'] in not_single):
				temp = index[i]
			else:
				temp = up_down_complement(index[i] , 'down')
			list_insert(ans_list,pos1[-1],temp)
		else:
			if (len(index[i]) == 1) and (df.loc[df.index[index[i][0]],'name'] in not_single):
				temp = index[i]
			else:
				temp = up_down_complement(index[i] , 'up')
			list_insert(ans_list,pos2[-1],temp)
	return ans_list

def sum_rebuil(df,pic_shape = (1024,1024)):
	#求和的重构
	sum_may_index = np.where(df['name'] == r'\sum')[0]
	if (len(sum_may_index) == 0):#没有分号可能
		return df
	while (len(sum_may_index)):
		m = get_grahp_matrix(df)
		index = sum_may_index[0]

		up = np.where(m[index,:] == 8)[0]
		down = np.where(m[index,:] == 4)[0]
		up_list = df.index[up]
		down_list = df.index[down]
		up_ans = rebuil(df.loc[up_list,:])
		down_ans = rebuil(df.loc[down_list,:])
		if (down_ans == '' and up_ans == ''):
			ans = r'\sum '
		else:
			ans = r'\sum_{' + down_ans + '}^{' + up_ans + '}'
		relevance = [df.index[index]] + list(up_list) + list(down_list)
		# x_list = df.loc[relevance,['xhigh','xlow','xmax','xmin','x']].values
		# y_list = df.loc[relevance,['ymax','ymin','y']].values
		df.loc[df.index[index]] = [ans] + get_relevance_attribute(df,relevance)
		df.drop(relevance[1:],inplace = True)
		df = sort_df(df)
		df = df.reset_index(drop = True)
		sum_may_index = np.where(df['name'] == r'\sum')[0]
	df = df.reset_index(drop = True)
	return df

def lim_rebuil(df,pic_shape = (1024,1024)):
	#求和的重构
	lim_may_index = np.where(df['name'] == r'\lim')[0]
	if (len(lim_may_index) == 0):#没有分号可能
		return df
	while (len(lim_may_index)):
		m = get_grahp_matrix(df)
		index = lim_may_index[0]

		down = np.where(m[index,:] == 4)[0]
		down_list = df.index[down]
		down_ans = rebuil(df.loc[down_list,:])

		ans = r'\lim_{' + down_ans + '}'
		relevance = [df.index[index]] + list(down_list)
		df.loc[df.index[index]] = [ans] + get_relevance_attribute(df,relevance)
		df.drop(relevance[1:],inplace = True)
		df = sort_df(df)
		df = df.reset_index(drop = True)
		lim_may_index = np.where(df['name'] == r'\lim')[0]
	df = df.reset_index(drop = True)
	return df

def int_rebuil(df,pic_shape = (1024,1024)):
	#积分的重构
	int_may_index = np.where(df['name'] == r'\int')[0]
	if (len(int_may_index) == 0):#没有积分的可能
		return df
	while (len(int_may_index)):
		index = df.index[int_may_index[0]]

		up = df.loc[index,'ymax'] - (df.loc[index,'ymax'] - df.loc[index,'ymin'])/3
		down = df.loc[index,'ymax'] - (df.loc[index,'ymax'] - df.loc[index,'ymin'])/3*2
		right = df[np.logical_not((df['ymin'] > up) | (df['ymax'] < down)) & (df['xmin'] >= df.loc[index,'xmax'])]
		d_index = np.where(df['name'] == r'd')[0]
		if (len(d_index) != 0 and len(df.index) >= (d_index[0]+2)):
			d_index = d_index[0]
			df.loc[df.index[d_index],'ymax'] = df.loc[df.index[d_index+1],'ymax']
			df.loc[df.index[d_index],'ymin'] = df.loc[df.index[d_index+1],'ymin']
			df.loc[df.index[d_index],'y'] = df.loc[df.index[d_index+1],'y']
		else:
			warnings.warn('未找到积分变量,识别过程可能存在错误！')
		right = right['xmin'].values
		right = np.min(right)

		up = df[(df['y'] >= up) & (df.loc[index,'xmax'] <= df['x']) & (df['x'] <= right)]
		down = df[(df['y'] <= down) & (df.loc[index,'xmax'] <= df['x']) & (df['x'] <= right)]
		up_ans = rebuil(up)
		down_ans = rebuil(down)
		if (up_ans == '' and down_ans == ''):
			ans = r'\int '
		else:
			ans = r'\int^{' + up_ans + r'}_{' + down_ans + r'}'
		relevance = [index] + list(up.index) + list(down.index)
		df.loc[df.index[index]] = [ans] + get_relevance_attribute(df,relevance)
		df.drop(relevance[1:],inplace = True)
		df = sort_df(df)
		df = df.reset_index(drop = True)
		int_may_index = np.where(df['name'] == r'\int')[0]
	df = df.reset_index(drop = True)
	return df

def expand(df,pic_shape = (1024,1024)):
	#用于符号边框的扩展以提高识别准确率
	df = frac_like_expand(df,pic_shape)
	# df = frac_expand(df,pic_shape)
	return df

def bracket_first(df,pic_shape = (1024,1024)):
	left = list(np.where(df['name'] == r'(')[0])
	right = list(np.where(df['name'] == r')')[0])
	if (len(left) == 0 or len(right) == 0):
		return df
	elif (len(left) != len(right)):
		warnings.warn('括号数量不匹配,优先重构模块关闭,识别过程可能存在错误!')
		return df
	else:
		while(len(left) != 0):
			m = np.zeros([len(left),len(right)])
			for i in range(len(left)):
				for j in range(len(right)):
					m[i,j] = right[j]-left[i]
			m[m <= 0] = np.inf
			left_index,right_index = np.where(m == m.min())
			start,end = left[left_index[0]],right[right_index[0]]
			up_ylow = max(df.loc[df.index[start],'ymax'],df.loc[df.index[end],'ymax'])
			down_yhigh = min(df.loc[df.index[start],'ymin'],df.loc[df.index[end],'ymin'])
			relevance = df[np.logical_not((df['ymin'] > up_ylow) | (df['ymax'] < down_yhigh)) &
			(df['xmin'] > df.loc[df.index[start],'xmin']) & (df['xmax'] < df.loc[df.index[end],'xmax'])]
			ans = '(' + rebuil(relevance) + ')'
			relevance = relevance.index
			relevance = [df.index[start]] + list(relevance) + [df.index[end]]
			df.loc[df.index[start]] = [ans] + get_relevance_attribute(df,relevance)
			df.drop(relevance[1:],inplace = True)
			df = sort_df(df)
			df = df.reset_index(drop = True)
			left = list(np.where(df['name'] == r'(')[0])
			right = list(np.where(df['name'] == r')')[0])

	return df

def order_control(df,pic_shape = (1024,1024)):
	#用于控制重构函数的调用顺序
	df = frac_rebuil(df,pic_shape)
	df = int_rebuil(df,pic_shape)
	df = sqrt_rebuil(df,pic_shape)
	df = lim_rebuil(df,pic_shape)
	df = sum_rebuil(df,pic_shape)
	df = sort_df(df)
	ans_list = up_down_rebuil(df,pic_shape)
	#再把整个列表中的标记转换为字符串
	ans = list_to_nameid(df,ans_list)
	return ans

def rebuil(df):
	if (len(df) == 0):
		warnings.warn('位置文件为空!', UserWarning)
		return ''
	df = df.copy()
	df = match_csv(df)
	pic_shape = get_pic_shape(df)
	# df = sort_df(df)
	# df = bracket_first(df,pic_shape)
	# df = expand(df,pic_shape)
	ans = order_control(df,pic_shape)
	# ans = semantics(ans)
	return ans

def finall_do(string,del_label_space = True ,latex_mode = True):
	#尾处理:
	#		去除上下标标记前面的空格
	#		两端数学符号标记
	label = [r'\_',r'\^',r'\(',r'\)',r'\[',r'\]',r'\{',r'\}']#识别的标识符
	replace_label = [r'_',r'^',r'(',r')',r'[',r']',r'{',r'}']#需要替换成的标识符
	# label = ['_','\^','\(','\)','\[','\]','\{','\}']
	string = semantics(string)
	if (del_label_space):
		for i in range(len(label)):
			string = re.sub(r'\s'+label[i],replace_label[i],string)
	if (latex_mode):
		string = r'$$' + string + r'$$'
	return string

'''
#--------------------------------------------------------------------------------------------------#
以下为语义逻辑识别
'''
# from functools import reduce
sym=['\\in','=','+','\\times','\\div','\\int','\\rightarrow','\\neq','>','<','\\geq','\\ge','\\ldots','\\pm','\\leq',']','[','\\exists',',','-','\\forall']
#一元函数二元函数传入值的匹配
one_var_fun = [r'\sin',r'\cos',r'\tan']
two_var_fun = [r'\log']

def fun_var_match(string):
	##一元函数二元函数传入值的匹配
	for i in one_var_fun:
		patter = re.compile('\\'+i)
		start = 0
		while (1):
			temp = patter.search(string[start:])
			if (temp == None):
				break
			start += temp.end()
			for j,k in enumerate(string[start:]):
				if (k == r' '):
					pass
				elif (k in sym):
					break
			end = start + j
			sub_string = string[start:end]
			temp,j,mid = [],0,''
			while (j < len(sub_string)):
				if (sub_string[j] == r'^' or sub_string[j] == r'_'):
					temp.append((j,sub_string[j]))
					while(j < len(sub_string) and sub_string[j] != r'}'):
						j += 1
				elif (sub_string[j] == r' '):
					pass
				else:
					mid += sub_string[j]
					break
				j += 1
			if (mid == ''):#这表明出现预期情况
				if (len(temp) == 1):
					sub_string = sub_string[:temp[0][0]] + sub_string[temp[0][0]+1:-1]
				else:
					pos1 = temp[1][0]
					pos2 = pos1+1
					while(pos2 < len(sub_string) and sub_string[pos2] != r'}'):
						pos2 += 1
					sub_string = sub_string[:pos1] + sub_string[pos1+2:pos2] + sub_string[pos2+1:]
			# print(string[end:])
			temp = len(string[:start] + sub_string)
			string = string[:start] + sub_string + string[end:]
			start = temp+1
	return string

def up_down_num(string):
	#给定一个字符串,返回其上下标的数量
	#上标一个加1
	#下标一个减1
	num = []
	for i in string:
		if (i == r'^{'):
			num.append(1)
		elif (i == r'_{'):
			num.append(-1)
		elif (i == r'}'):
			num.pop()
	return reduce(lambda x,y:x+y,num)

def int_match(string):
	return string

def semantics(string):
	#根据语义对答案进行修正
	string = fun_var_match(string)
	return string


'''
#--------------------------------------------------------------------------------------------------#
以下为异常处理
'''

csv_order = ['name','x','y','xmax','xmin','ymax','ymin']
def match_csv(df):
	if (list(df.columns) == csv_order):
		return sort_df(df)
	if ('xmid' in list(df.columns)):
		df.rename(columns = {'xmid':'x','ymid':'y'},inplace = True)
	df = df[csv_order]
	return sort_df(df)

def get_pic_shape(data):
	width = data['xmax'].max()-data['xmin'].min()
	height = data['ymax'].max()-data['ymin'].min()
	return (width,height)

