#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import struct

# normalizeモジュールを使用する
import normalize as norm


parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# 各種パラメータ
batchsize = 10
n_epoch   = 10
h_units   = 100
in_units  = 1
out_units = 1
floatsize = 4

# データ数
N_all = 110000
N_train = 100000
N_test = 10000

# 正規化用(入力系列の最大値・最小値の設定)
maxi = 500
mini = 0


# ファイル名
train_inname = "train_x.float"
train_outname = "train_y.float"
test_inname = "test_x.float"

train_loss_fname = "train_loss.txt"
result_fname = "result.txt"


# 入力データ(トレーニング用)読み込み
print 'fetch train dataset (infile)'
fin = open(train_inname, "rb")

indatalist = []

# リストの中にリストを作る
while 1:
    tmplist = []
    for i in range(in_units):
        # Read binary file (recognized as string)
        b = fin.read(floatsize)
        # 終端まで読み込む
        if b == "":
            break        
        # 文字列からフロート型に変換する(出力はタプル)
        tmp = struct.unpack("f", b)
        # 正規化
        work = norm.normalize(tmp[0], maxi, mini)
        
        tmplist.append(work)
    if b == "":
        break
    else:
        indatalist.append(tmplist)

# 出力データ(トレーニング用)読み込み
outdatalist = []
print 'fetch train dataset (outflie)'
fout = open(train_outname, "rb")
while 1:
    tmplist = []
    for i in range(out_units):
        # Read binary file (recognized as string)
        b = fout.read(floatsize)
        # 終端まで読み込む
        if b == "":
            break        
        # 文字列からフロート型に変換する(出力はタプル)
        tmp = struct.unpack("f", b)
        # 1～5を0～1に正規化
        #work = norm.normalize(tmp[0], 5, 1)

        #tmplist.append(work)
        tmplist.append(tmp[0])
    if b == "":
        break
    else:
        outdatalist.append(tmplist)

# 入力データ(テスト用)読み込み
print 'fetch test dataset (infile)'
fin_test = open(test_inname, "rb")
testdatalist = []

# リストの中にリストを作る
while 1:
    tmplist = []
    for i in range(in_units):
        # Read binary file (recognized as string)
        b = fin_test.read(floatsize)
        # 終端まで読み込む
        if b == "":
            break        
        # 文字列からフロート型に変換する(出力はタプル)
        tmp = struct.unpack("f", b)

        # 正規化
        work = norm.normalize(tmp[0], maxi, mini)
        tmplist.append(work)
    if b == "":
        break
    else:
        testdatalist.append(tmplist)



# データの格納(入力データはどちらも正規化されたもの)
x_train = np.array(indatalist)
y_train = np.array(outdatalist)
x_test = np.array(testdatalist)


# Prepare multi-layer perceptron model
model = FunctionSet(l1=F.Linear(in_units, h_units),
                    l2=F.Linear(h_units, h_units),
                    l3=F.Linear(h_units, out_units))


# Neural net architecture
def train_forward(x_data, y_data):
    # Variable(chainer独自の型)に変換
    x, t = Variable(x_data), Variable(y_data)
    # 第三引数がFalse場合は第一引数の値をそのまま返す(trainの場合のみDropoutを行い、testの場合は行わないようにする)
    # hは前の層からの出力
    h1 = F.dropout(F.relu(model.l1(x)))
    h2 = F.dropout(F.relu(model.l2(h1)))
    y  = model.l3(h2)

    # 2乗平均誤差(MSE)を返す
    return F.mean_squared_error(y, t)

def test_forward(x_data):
    # Variable(chainer独自の型)に変換
    x = Variable(x_data)
    # 第三引数がFalse場合は第一引数の値をそのまま返す(trainの場合のみDropoutを行い、testの場合は行わないようにする)
    # hは前の層からの出力
    h1 = F.relu(model.l1(x))
    h2 = F.relu(model.l2(h1))
    y  = model.l3(h2)

    # 出力データを返す
    return y

# Setup optimizer(BPの際に勾配を計算するアルゴリズム)
optimizer = optimizers.SGD()
optimizer.setup(model.collect_parameters())



#########################################################
# Learning loop
train_losstext = ""
test_losstext = ""
for epoch in xrange(1, n_epoch+1):
    print 'epoch', epoch

    # Training
    # 0～N-1までの値をランダムな順番でarrayに格納
    perm = np.random.permutation(N_train)
    sum_loss = 0

    # バッチ処理
    for i in xrange(0, N_train, batchsize):
        # print perm[i:i+batchsize]
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]

        
        # 勾配を0に初期化
        optimizer.zero_grads()

        # NNへ入力
        loss = train_forward(x_batch, y_batch)
        # 逆誤差伝搬
        loss.backward()
        # 勾配を更新
        optimizer.update()
        # もしGPUを使っていたら、ArrayをCPU対応のものに変換する
        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize

    train_losstext += "epoch" + str(epoch) + "\t" + str(sum_loss / N_train) + "\n"


    # Evaluation
    # 出力結果の格納
    out = test_forward(x_test).data
      

print "\n\n-----------------result-----------------------\n"

f_train_loss = open(train_loss_fname, "w")
f_train_loss.write(train_losstext)
print "MSE in training step\n",train_losstext, "\n\n"

f_result = open(result_fname, "w")
for i in range(N_test):
    # リストから取り出すために[0]をつける
    f_result.write(str(out[i][0]) + "\n")
    # print str(out[i][0]) + ","



