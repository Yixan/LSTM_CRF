# coding: utf-8
import tensorflow as tf
from tensorflow.contrib.crf import viterbi_decode
import lstm_crf_data_helper, lstm_crf_model
from util import fileUtil
import os
'''定义常量'''
MODEL_SAVE_PATH='mymodel/'
MODEL_NAME='checkpoint5-25'
tf.app.flags.DEFINE_string("f","","test")
tf.flags.DEFINE_integer("embeddings_size", 300, "每个字向量的维度")
tf.flags.DEFINE_integer("hidden_dim", 300, "LSTM隐藏层细胞的个数")
tf.flags.DEFINE_integer("batch_size", 128, "每个批次的大小")
tf.flags.DEFINE_integer("num_epochs", 2, "训练的轮数")
tf.flags.DEFINE_float("keep_prob", 0.5, "丢失率")
tf.flags.DEFINE_float("forget_bias", 0.8, "遗忘率")
tf.flags.DEFINE_float("clip_grad", 5.0, "梯度的范围")
tf.flags.DEFINE_float("learning_rate", 0.001, "学习率")
tf.app.flags.DEFINE_string("train_dir","log",'储存路径')
FLAGS = tf.flags.FLAGS
# tag2label = {'0':0,'B':1,'M':2,'E':3,'S':4}
# label2tag = {0:'0',1:'B',2:'M',3:'E',4:'S'}
tag2label=lstm_crf_data_helper.get_tag2label('D:\\nlp\\LSTM_CRF\\data\\pos_tag2num_dict.pkl')
label2tag=lstm_crf_data_helper.get_tag2label('D:\\nlp\\LSTM_CRF\\data\\pos_num2tag_dict.pkl')
x = tf.placeholder(tf.int32, [None, None], name='input')
y = tf.placeholder(tf.int32, [None, None], name='output')
#每一句的长度
sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
#丢失率
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
# 获得字与字索引映射的向量
# word2id_dict = lstm_crf_data_helper.get_word_id('data\\char2num.pkl')
word2id_dict = lstm_crf_data_helper.get_word_id('data\\pos.pkl')
# word2id_dict = lstm_crf_data_helper.get_word_id('data\\char2num.pkl')
# 获得总的标注类别数
num_tags = len(tag2label)
'''处理数据'''
def get_data(file_location):
    # 获得数据
    # sentences_list, tags_list是没有按照最大长度填充的标签
    sentences_list, tags_list = fileUtil.get_data(file_location)
    # 完成tag向索引的映射
    tags_id_list = lstm_crf_data_helper.tags2id(tags_list, tag2label)
    # 对索引进行填充
    labels, _ = lstm_crf_data_helper.padding_sentences(tags_id_list)
    # 获得句子中每个字的id
    sentences_id_list = lstm_crf_data_helper.sentence2id(sentences_list, word2id_dict)
    # 对句子或标注序列索引进行填充并获得每个句子的长度
    sen_index_list, sen_len_list = lstm_crf_data_helper.padding_sentences(sentences_id_list)
    return sen_index_list, labels, sen_len_list, tags_list

'''得到参考的标签'''
def get_refer_tag(referfile):
    refer_list=[]
    with open(referfile,'r',encoding = 'UTF-8') as f:
        lines=f.readlines()
        for line in lines:
            line=line.strip()
            words = line.split()
            refer_list.append(words)
    return refer_list

'''标签转化成词语'''
def label2word(label_list,model='val'):
    if model=='val':
        refer_file='val.txt'
    else:
        refer_file='test.txt'
    tags_list = []
    for labels in label_list:
        tags = []
        for i in labels:
            tags.append(label2tag[i])
        tags_list.append(tags)
    refer_list=[]
    with open(refer_file,'r',encoding='utf-8') as file:
        lines=file.readlines()
        for line in lines:
            words=line.split()
            for word in words:
#                 print(word.split("##")[0])
                refer_list.append(word.split("##")[0])
    i=0
    word=''
    sentences=[]
    sentences_list=[]
    for tags in tags_list:
        for tag in tags:
            if tag=='S':
                word=refer_list[i]
                sentences.append(word)
                word=''
            elif tag=='B':
                word = refer_list[i]
            elif tag=='M':
                word = refer_list[i]+word
            elif tag=='E':
                word=refer_list[i]+word
                sentences.append(word)
                word = ''
            i=i+1
            sentences=[]
        sentences_list.append(sentences)
    return sentences_list

test_words_list=get_refer_tag('test_cws1.txt')
val_words_list=get_refer_tag('val_cws.txt')
train_sen_index_list, train_labels, train_sen_len_list, _ = get_data('train.txt')
val_sen_index_list, val_labels, val_sen_len_list, val_tags_list=get_data('val.txt')
test_sen_index_list, test_labels, test_sen_len_list, test_tags_list = get_data('test.txt')

'''计算F'''
def evalute(sentences_list,refer_list):
    count_right = 0
    count_split = 0
    count_gold = 0
    for i,sentence in enumerate(sentences_list):
        # sentence = sentence.strip()
        # goldlist = sentence.split()
        count_gold += len(sentence)
        tmp_gold = sentence
        line2 =refer_list[i]
        # line2 = line2.strip()
        inlist = line2
        count_split += len(inlist)
        tmp_in = inlist
        for i,key in enumerate(tmp_in):
            if key in tmp_gold:
                count_right += 1
                tmp_gold.remove(key)

    P = count_right / count_split
    R = count_right / count_gold
    return  2 * P * R / (P + R)

'''解码得到句子'''
def decode(logits, transition_params,model='test'):
    if model=='test':
        words_list=test_words_list
    else:
        words_list=val_words_list
    # 对测试集进行测试
    label_list = []
    for logit, seq_len in zip(logits, test_sen_len_list):
        # viterbi_decode通俗一点,作用就是返回最好的标签序列.这个函数只能够在测试时使用,在tensorflow外部解码
        # viterbi: 一个形状为[seq_len] 显示了最高分的标签索引的列表.
        # viterbi_score: 序列对应的概率值
        # 这是解码的过程，利用维比特算法结合概率转移矩阵求得最大的可能标注概率
        viterbi_seq, _ = viterbi_decode(logit[:seq_len], transition_params)
        label_list.append(viterbi_seq)
    sentences_list = label2word(label_list,model=model)
    # F = evalute(sentences_list, words_list)
    return sentences_list,words_list

'''生成CSV'''
def write2CSV(para_dict):
    from pandas import DataFrame
    df = DataFrame(para_dict)
    df.to_csv("paraments.csv",index=False,sep=',')





'''准备训练所需变量以及训练模型'''
def train():
    losslist=[]
    # 首先是embedding层获得词向量数据
    with tf.name_scope("embedding"):
        #构造词嵌入矩阵，每个词向量维度为300维，随机产生
        embedding_mat = lstm_crf_data_helper.random_embedding(word2id_dict, FLAGS.embeddings_size)
        print(len(embedding_mat))
        #选取此嵌入矩阵里索引对应的元素 取对应字的字向量
        input_x = tf.nn.embedding_lookup(embedding_mat, x)
        # input_x=tf.nn.dropout(input_x,keep_prob=FLAGS.keep_prob)
    #构建神经网络
    BiLSTM_CRF = lstm_crf_model.BiLSTM_CRF(FLAGS.hidden_dim, num_tags, input_x,
                                           sequence_lengths, keep_prob, y,FLAGS.forget_bias)
    #前向传播，loss:损失函数值,transition_params是:RF的转换矩阵,logits是预测值[seq_len，num_tags]
    loss, transition_params, logits = BiLSTM_CRF.positive_propagation()
    #全局步数
    global_step = tf.Variable(0, name="global_step", trainable=False)
    #自适应学习率
    # learning_rate=tf.train.exponential_decay(3.0, global_step, 3, 0.3, staircase=True)
    #优化器
    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    # optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # 计算梯度，返回的是：A list of (gradient, variable) pairs
    grads_and_vars = optim.compute_gradients(loss)
    #控制梯度范围，还不知道g,v的具体含义
    grads_and_vars_clip = [[tf.clip_by_value(g, -FLAGS.clip_grad, FLAGS.clip_grad), v] for g, v in grads_and_vars]
    #将计算出的梯度应用到变量上，是函数minimize()的第二部分，返回一个应用指定的梯度的操作Operation，对global_step做自增操作
    train_op = optim.apply_gradients(grads_and_vars_clip, global_step=global_step)
    #断点续训
    saver = tf.train.Saver()
    # tensorboard可视化
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    merged_summary_op = tf.summary.merge_all()
    losslist = []
    test_f_list = []
    val_f_list = []
    g_list=[]
    v_list=[]
    lr_list=[]
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        # 模型的训练
        num_inter = int(len(train_sen_len_list) / FLAGS.batch_size)
        # 训练epoch轮
        step = 0
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)  # 获取checkpoints对象
        if ckpt and ckpt.model_checkpoint_path:  ##判断ckpt是否为空，若不为空，才进行模型的加载，否则从头开始训练
            saver.restore(sess, ckpt.model_checkpoint_path)  # 恢复保存的神经网络结构，实现断点续训
        for epoch in range(FLAGS.num_epochs):
            # 每一轮中分batch训练
            for i in range(num_inter):
                start = i * FLAGS.batch_size
                end = (i + 1) * FLAGS.batch_size
                feed_dict = {x: train_sen_index_list[start:end], y: train_labels[start:end],
                             sequence_lengths: train_sen_len_list[start:end], keep_prob: FLAGS.keep_prob}
                train_loss,train_op = sess.run([loss,train_op],feed_dict=feed_dict)
                losslist.append(train_loss)
                # 验证集准确率
                logits2, transition_params2  = sess.run([logits, transition_params], feed_dict={
                    x: val_sen_index_list, y: val_labels,
                    sequence_lengths: val_sen_len_list, keep_prob: 1.0
                })
                sentences_list, words_list = decode(logits2, transition_params2, model='val')
                F = evalute(sentences_list, words_list)
                val_f_list.append(F)
                # print("验证集F的值为：" + F)
                # 测试集准确率
                logits1, transition_params1 = sess.run([logits, transition_params], feed_dict={
                    x: test_sen_index_list, y: test_labels,
                    sequence_lengths: test_sen_len_list, keep_prob: 1.0
                })
                sentences_list, words_list = decode(logits1, transition_params1)
                F = evalute(sentences_list, words_list)
                test_f_list.append(F)
                print("测试集F的值为{:.4f}：" .format(F))
                # if i % 10 == 0:
                # 算loss
                # train_loss = sess.run(loss,
                #                       feed_dict={x: train_sen_index_list[start:end], y: train_labels[start:end],
                #                                  sequence_lengths: train_sen_len_list[start:end],
                #                                  keep_prob: FLAGS.keep_prob})

                merged_summary = tf.summary.merge_all()
                summary = sess.run(merged_summary_op,
                                   feed_dict={x: train_sen_index_list[start:end], y: train_labels[start:end],
                                              sequence_lengths: train_sen_len_list[start:end],
                                              keep_prob: FLAGS.keep_prob})
                writer.add_summary(summary, step)
                step = step + 1
                print("总步数:", step)
                print("epoch:%d step:%d loss is:%s" % (epoch + 1, i, train_loss))
                # 更新变量不算loss
                # sess.run(train_op, feed_dict=feed_dict)
            # if val_f_list[-1]<val_f_list[-2]:
            #     print("提前终止，轮数%d"%(epoch+1))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            writer.close()

    para_dict={'loss':losslist,'test_f':test_f_list}
    write2CSV(para_dict)

if __name__=='__main__':
    train()