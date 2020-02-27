import numpy as np
import pandas as pd
lgl=['\\in','=','+','\\times','\\div','\\int','\\rightarrow','\\neq','>','<','\\geq','\\ge','\\ldots','\\pm','\\leq',']','[','\\exists','(',')',',','-','^','_','\\forall']

def replace(string,berep,rep):
	if type(berep) != list:
		return string.replace(berep,rep)
	else:
		for i in berep:
			string = string.replace(i,rep)
	return string


# function wer_score:
"""
Calculation of WER with Levenshtein distance.

Works only for iterables up to 254 elements (uint8).
O(nm) time ans space complexity.

Parameters
----------
r : list
h : list

Returns
-------
int

Examples
--------
>>> wer("who is there".split(), "is there".split())
1
>>> wer("who is there".split(), "".split())
3
>>> wer("".split(), "who is there".split())
3
"""
def wer_score(r, h):
	d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
	d = d.reshape((len(r)+1, len(h)+1))
	for i in range(len(r)+1):
		for j in range(len(h)+1):
			if i == 0:
				d[0][j] = j
			elif j == 0:
				d[i][0] = i

	# computation
	for i in range(1, len(r)+1):
		for j in range(1, len(h)+1):
			if r[i-1] == h[j-1]:
				d[i][j] = d[i-1][j-1]
			else:
				substitution = d[i-1][j-1] + 1
				insertion    = d[i][j-1] + 1
				deletion     = d[i-1][j] + 1
				d[i][j] = min(substitution, insertion, deletion)

	return d[len(r)][len(h)]

#输入:正确的标签字符串,预测的字符串序列
#输出:该单词的WER
def WER(ytest,ypre):
	num = len(ytest.split())
	if ypre == []:
		return 1
	#先把序列中的'\','{','}'去掉
	ytest = replace(ytest,['{','}'],' ')
	ypre = replace(ypre,['{','}'],' ')
	for i in lgl:
		ytest = ytest.replace(i,' '+i+' ')
		ypre = ypre.replace(i,' '+i+' ')
	# ytest = [i for i in filter(lambda x:x!='',ytest.split(' '))]
	# ypre = [i for i in filter(lambda x:x!='',ypre.split(' '))]
	return wer_score(ytest.split(),ypre.split())/num
