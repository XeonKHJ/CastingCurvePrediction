# Process original dataset file into a input dataset files.
# Original dataset file look like this:
# "LV_TRG#2 Time";"LV_TRG#2 ValueY";"LV_ACT#2 Time";"LV_ACT#2 ValueY";"STP_POS#2 Time";"STP_POS#2 ValueY";"tundish_weight2 Time";"tundish_weight2 ValueY";"CST_SPD#2 Time";"CST_SPD#2 ValueY";"Auto_SPD_SET2 Time";"Auto_SPD_SET2 ValueY";"CURRENT2 Time";"CURRENT2 ValueY";"sensor_fail Time";"sensor_fail ValueY"
# 2021/4/7 17:54:14;80;2021/4/7 17:54:14;0.16120021045208;2021/4/7 17:54:14;-0.244000017642975;2021/4/7 17:54:14;17.0464401245117;2021/4/7 17:54:14;0;;;2021/4/7 17:54:14;0;;
# 2021/4/7 17:54:14;80;2021/4/7 17:54:14;0.157102972269058;2021/4/7 17:54:14;-0.238000005483627;2021/4/7 17:54:14;17.2254791259766;2021/4/7 17:54:14;0;;;2021/4/7 17:54:14;0;;
# 2021/4/7 17:54:15;80;2021/4/7 17:54:15;0.157499492168427;2021/4/7 17:54:15;-0.220000013709068;2021/4/7 17:54:15;17.2960052490234;2021/4/7 17:54:15;0;;;2021/4/7 17:54:15;0;;

# input dataset files look like this:
# value0,value1,value2,value3,value4,value5,value6
# -0.227000012993813,0.554000020027161,-0.396000027656555,-2.1470000743866,1.28600001335144,-0.512000024318695,0.0
# -0.198000013828278,0.554000020027161,-0.396000027656555,-2.14800000190735,1.27700006961823,-0.435000032186508,0.0
# -0.176000013947487,0.554000020027161,-0.396000027656555,-2.1470000743866,1.27700006961823,-0.41100001335144,0.0
# -0.176000013947487,0.554000020027161,-0.397000014781952,-2.14600014686584,1.27600002288818,-0.367000013589859,0.0
# -0.176000013947487,0.555000007152557,-0.397000014781952,-2.14500021934509,1.27500009536743,-0.336000025272369,0.0
# -0.175000011920929,0.554000020027161,-0.396000027656555,-2.14500021934509,1.27500009536743,-0.235000014305115,0.0
# -0.175000011920929,0.555000007152557,-0.396000027656555,-2.14500021934509,1.27400004863739,-0.155000001192093,0.0
# -0.175000011920929,0.553000032901764,-0.396000027656555,-2.1470000743866,1.27400004863739,-0.112000003457069,0.00100000016391277
# -0.175000011920929,0.554000020027161,-0.396000027656555,-5.22900009155273,1.27400004863739,-0.0160000007599592,0.0
# NOTE: values from value[x] is the values from dataset[x].
# this process extracts needed info from dataset files and gather them into one single file.



import os
import sys
from typing import Counter, ForwardRef
import numpy
from numpy.lib.function_base import append
from numpy.lib.npyio import fromregex
import pandas

allfiles = 0
def PreReadData():
    datasetFolder = "../../../datasets/Datas/"
    allfiles = os.listdir(datasetFolder)

    allStage = list()
    
    for file in allfiles:
        stage1f = list()
        stage2f = list()
        datafile = open(datasetFolder + file, encoding='utf-16')
        lines = datafile.readlines()

        indexToStart = 1
        while float(lines[indexToStart].split(';')[5]) == 0:
            indexToStart += 1

        attrsList = list() 
        for i in range(lines.__len__() - indexToStart):
            if i == 0:
                continue
            line = lines[i + indexToStart]
            attris = line.split(';')
            attrsList.append(float(attris[5]))
        allStage.append(attrsList)
    return allStage

# return a array that contains following info:
# smallestComp[dataIndex] = index of ogData that has smallest MSE with smallestComp[dataIndex]
def MoveFowardBy(data,ogData,step):
    ogLen = ogData.__len__()
    len = data.__len__()
    smallestComp = list()
    for i in range (int(ogLen/step)):
        smallestComp.append(0.0)

    if step > 0:
        for i in range(0, len, step):
            sublist2 = data[i:step+i]
            minMse = sys.float_info.max
            minIndex = int(len/step)
            for j in range(0, ogLen, step):
                ogSublist = ogData[j:(step + j)]
                mse = mseData(sublist2, ogSublist)
                if mse < minMse:
                    minMse = mse
                    minIndex = int(j/step)           
            smallestComp[int(i/step)] = list()
            smallestComp[int(i/step)].append(minIndex)  
            smallestComp[int(i/step)].append(minMse)
    return smallestComp
            
            
           
def mseData(data1, data2):
    data1Np = numpy.array(data1)
    data2Np = numpy.array(data2)
    mse = (numpy.square(data1Np - data2Np)).mean()
    return mse

def writeFile(dataset):
    datasetDict = dict()
    for i in range(len(dataset)):
        datasetDict[('value'+str(i))] = dataset[i]
    dataframe = pandas.DataFrame(datasetDict)
    dataframe.to_csv("test.csv",index=False,sep=',')


datasetLength = 2500

step = 100
dataset = PreReadData()
startindice = list()
for data in dataset:
    for dataIndice in range(len(data)):
        if data[dataIndice] > 5:
            startindice.append(dataIndice)
            break

for i in range(len(dataset)):
    dataset[i] = dataset[i][startindice[i] - 50: startindice[i] + datasetLength - 50]



# minMse = 0
writeFile(dataset)

print(dataset)
