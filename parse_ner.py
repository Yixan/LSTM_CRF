# -*- coding: utf-8 -*-

import re
# [平均 [红细胞 血红蛋白]bod 含量]tes
# 平均 [红细胞 血红蛋白]bod 含量
ner_list=list()
# def parse_ner(ner,tag):
#     stack=0
#     ner1=''
#     inner=''
#     remained=''
#     if '[' in ner:
#         for i, char in enumerate(ner):
#             if char == '[':
#                 stack += 1
#             if stack > 0:
#                 ner1 = ner1 + char
#             elif stack == 0:
#                 remained=remained+char
#             if char == ']':
#                 stack -= 1
#                 if stack == 0 and ner != '':
#                     remained = remained+'##'
#                     tag = ner[i + 1:i + 4]
#                     inner=parse_ner(ner1[1:-1],tag)
#     else:
#         inner=ner
#         inner_list=[inner,'end',tag]
#         # ner_list.append([inner,tag])
#         return inner_list
#     ner_list.append([remained.replace(tag,'',1), inner, tag])

# parse_ner('排尿性 [膀胱 尿道]bod 造影','tes')
# print(ner_list)
innerner=''
def parse_ner_str(ner):
    stack=0
    ner1=''
    inner=''
    remained=''
    tag=''
    if '[' in ner:
        for i, char in enumerate(ner):
            if char == '[':
                stack += 1
            if stack > 0:
                ner1 = ner1 + char
            elif stack == 0:
                remained=remained+char
            if char == ']':
                stack -= 1
                if stack == 0 and ner != '':
                    inner=parse_ner_str(ner1[1:-1])
                    tag = ner[i + 1:i + 4]
                    remained = (remained+inner).replace(tag, '', 1)

                    ner1=''
    else:
        return str(ner)
    return remained.replace(tag, '', 1)

# print(parse_ner_str('[[幽门]bod 内 液体 外溢 压迫 [胃 出口]bod]'))
def get_nerstr(words,tag):
    nerstr=''
    for word in words.split():
        # print(word)
        if len(word) == 1:
            nerstr =nerstr+word + " S-"+tag+'\n'
        else:
            nerstr =nerstr+word[0] + " B-"+tag+'\n'
            for w in word[1:len(word) - 1]:
                nerstr = nerstr + w + " M-" + tag + '\n'
            nerstr =nerstr+word[-1] + " E-"+tag+'\n'
    return nerstr

def write(item,savename):
    with open(savename, 'a', encoding='utf-8') as file:
        file.write(item)
def ner(filename,savename):
    with open(filename ,'r',encoding='utf-8') as file:
        lines=file.readlines()
        for j,line in enumerate(lines):
            # f.write(str(j) + ',')
            ner=''
            normal=''
            stack = 0
            for i,char in enumerate(line):
                if char =='[':
                    stack+=1
                if stack>0:
                    ner = ner + char
                elif re.search('\s+',char)==None:
                    normal=normal+char+' O\n'
                if char ==']':
                    stack -= 1
                    if stack==0 and ner!='':
                        # print(ner)
                        tag=line[i+1:i+4]
                        nerstr=get_nerstr(parse_ner_str(ner[1:-1]),tag)
                        # print(nerstr)
                        normal=normal+nerstr
                        # write_word(parse_ner_str(ner[1:-1]),tag)
                        # write(tag)
                        # f.write(',')
                        # f.write('##')
                        # f.write(tag)
                        # f.write(',')
                        # f.write('\n')
                        ner=''
            # f.write('\n')
            write(normal,savename)
            # write('\n')

testfile='D:\\nlp\\LSTM_CRF\\data\\testset1\\test_ner1.txt'
testout='D:\\nlp\\LSTM_CRF\\data\\testset1\\test_ner_tag.txt'
trainfile='D:\\nlp\\LSTM_CRF\\data\\trainset\\train_ner.txt'
trainout='D:\\nlp\\LSTM_CRF\\data\\trainset\\train_ner_tag.txt'
valfile='D:\\nlp\\LSTM_CRF\\data\\devset\\val_ner.txt'
varout='D:\\nlp\\LSTM_CRF\\data\\devset\\val_ner_tag.txt'
ner(testfile,testout)
ner(trainfile,trainout)
ner(valfile,varout)
# print(ner_list)

