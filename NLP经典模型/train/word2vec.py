import time
import pickle
from torch.utils.data import DataLoader

from tools import word2vec_trainer
from Qmatter.models.nlp import word2vec
from tools.word2vec_build_data import *

def train_cbow():
    #../..")))  # 返回上上个目录
    filepath = "../../../tools/text8.txt"
    #超参数的设置，包括
    window_size = 5    #上下文窗口
    embed_dim = 100    #词向量维度
    batch_size = 100   #批大小
    num_epochs = 10    #训练epoch
    neg_num = 5        #负样本数
    learning_rate = 1e-3 #学习率
    #记录开始训练时间，start
    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    outputs_dir = f"../outputs/cbow-{now_time}"
    os.makedirs(outputs_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 读取语料库
    corpus= load_text(filepath)
    # 语料库预处理
    corpus=word_preprocess(corpus)
    # 词汇标签化
    word2id_dict,word2freq_dict,id2word_dict=word_freq2id(corpus)
    corpus=corpus2id(corpus,word2id_dict)
    # 下采样
    corpus=subsampling(corpus,word2freq_dict)
    # 中心词及上下文配对选择
    contexts, targets = create_context_target(corpus, window_size)
    #计算语料库词汇总数
    vocab_size = len(word2id_dict)
    corpus_info = {
        "word2id": word2id_dict,
        "id2word": id2word_dict,
    }
    save_path = os.path.join(outputs_dir, "corpus_info.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(corpus_info, f)
    #负采样
    negative_sampler = NegativeSampler(word2id_dict,word2freq_dict,id2word_dict, neg_num)
    #利用重写的数据类加载数据集
    train_dataset = CBOWDataset(
        contexts=contexts,
        targets=targets,
        negative_sampler=negative_sampler,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.generate_batch,
        num_workers=0,
        pin_memory=True,
    )

    model = word2vec.CBOW(vocab_size, embed_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = word2vec_trainer.Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        outputs_dir=outputs_dir,
        num_epochs=num_epochs,
        device=device,
    )

    trainer.train()


def train_skipgram():
    filepath = "../../../tools/text8.txt"
    window_size = 5
    embed_dim = 100
    batch_size = 100
    num_epochs = 10
    negative_sample_size = 5
    learning_rate = 1e-3
    now_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    outputs_dir = f"../outputs/skipgram-{now_time}/"
    os.makedirs(outputs_dir, exist_ok=True)
    device = torch.device("cuda")

    corpus, word2id, id2word = load_text(filepath)
    contexts, targets = create_context_target(corpus, window_size)
    vocab_size = len(word2id)

    corpus_info = {
        "corpus": corpus,
        "word2id": word2id,
        "id2word": id2word,
        "contexts": contexts,
        "targets": targets,
    }

    with open("../../../tools/text8.txt", "wb") as f:
        pickle.dump(corpus_info, f)

    negative_sampler = NegativeSampler(corpus, negative_sample_size)

    train_dataset = SkipGramDataset(
        contexts=contexts,
        centers=targets,
        negative_sampler=negative_sampler,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.generate_batch,
        num_workers=0,
        pin_memory=True,
    )

    model = word2vec.SkipGram(vocab_size, embed_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # trainer = Trainer(
    #     model=model,
    #     optimizer=optimizer,
    #     train_dataloader=train_dataloader,
    #     outputs_dir=outputs_dir,
    #     num_epochs=num_epochs,
    #     device=device,
    # )
    #
    # trainer.train()


if __name__ == "__main__":
    os.chdir(sys.path[0])
    train_cbow()
   # train_skipgram()

# # 定义cbow训练网络结构
# # 一般来说，在使用nn训练的时候，我们需要通过一个类来定义网络结构，这个类继承了paddle.nn.Layer
# class CBOW(nn.Module):
#     def __init__(self, vocab_size, embedding_size, init_scale=0.1):
#         # vocab_size定义了这个CBOW这个模型的词表大小
#         # embedding_size定义了词向量的维度是多少
#         # init_scale定义了词向量初始化的范围，一般来说，比较小的初始化范围有助于模型训练
#         super(CBOW, self).__init__()
#         self.vocab_size = vocab_size
#         self.embedding_size = embedding_size
#
#         # 使用paddle.nn提供的Embedding函数，构造一个词向量参数
#         # 这个参数的大小为：self.vocab_size, self.embedding_size
#         # 这个参数的名称为：embedding_para
#         # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
#         self.embedding = nn.Embedding(
#             self.vocab_size,
#             self.embedding_size,
#             weight_attr=torch.nn.parameter(
#                 name='embedding_para',
#                 initializer=nn.initializer.Uniform(
#                     low=-0.5 / embedding_size, high=0.5 / embedding_size)))
#
#         # 使用paddle.nn提供的Embedding函数，构造另外一个词向量参数
#         # 这个参数的大小为：self.vocab_size, self.embedding_size
#         # 这个参数的名称为：embedding_para_out
#         # 这个参数的初始化方式为在[-init_scale, init_scale]区间进行均匀采样
#         # 跟上面不同的是，这个参数的名称跟上面不同，因此，
#         # embedding_para_out和embedding_para虽然有相同的shape，但是权重不共享
#         self.embedding_out =nn.Embedding(
#             self.vocab_size,
#             self.embedding_size,
#             weight_attr=torch.ParamAttr(
#                 name='embedding_out_para',
#                 initializer=nn.initializer.Uniform(
#                     low=-0.5 / embedding_size, high=0.5 / embedding_size)))
#
#     # 定义网络的前向计算逻辑
#     # center_words是一个tensor（mini-batch），表示中心词
#     # target_words是一个tensor（mini-batch），表示目标词
#     # label是一个tensor（mini-batch），表示这个词是正样本还是负样本（用0或1表示）
#     # 用于在训练中计算这个tensor中对应词的同义词，用于观察模型的训练效果
#     def forward(self, center_words, target_words, label):
#         # 首先，通过embedding_para（self.embedding）参数，将mini-batch中的词转换为词向量
#         # 这里center_words和eval_words_emb查询的是一个相同的参数
#         # 而target_words_emb查询的是另一个参数
#         center_words_emb = self.embedding(center_words)  # 上下文词
#         target_words_emb = self.embedding_out(target_words)  # 目标词
#
#         # center_words_emb = [batch_size, embedding_size]
#         # target_words_emb = [batch_size, embedding_size]
#         # 我们通过点乘的方式计算中心词到目标词的输出概率，并通过sigmoid函数估计这个词是正样本还是负样本的概率。
#         word_sim = torch.multiply(center_words_emb, target_words_emb)
#         word_sim = torch.sum(word_sim, axis=-1)
#         word_sim = torch.reshape(word_sim, shape=[-1])
#         pred = nn.functional.sigmoid(word_sim)
#
#         # 通过估计的输出概率定义损失函数，注意我们使用的是binary_cross_entropy函数
#         # 将sigmoid计算和cross entropy合并成一步计算可以更好的优化，所以输入的是word_sim，而不是pred
#
#         loss = nn.functional.binary_cross_entropy(nn.functional.sigmoid(word_sim), label)
#         loss = torch.mean(loss)
#
#         # 返回前向计算的结果，飞桨会通过backward函数自动计算出反向结果。
#         return pred, loss
# #定义一个使用word-embedding计算cos的函数
# def get_cos(query1_token, query2_token, embed):
#     W = embed
#     x = W[word2id_dict[query1_token]]
#     y = W[word2id_dict[query2_token]]
#     cos = np.dot(x, y) / np.sqrt(np.sum(y * y) * np.sum(x * x) + 1e-9)
#     flat = cos.flatten()
#     print("单词1 %s 和单词2 %s 的cos结果为 %f" %(query1_token, query2_token, cos) )
#
#
##
# if __name__=="__main__":
#   #  download()
#     corpus = load_text8()
#     # 打印前500个字符，简要看一下这个语料的样子
#     print(corpus[:500])
#     corpus = data_preprocess(corpus)
#     print(corpus[:50])
#     word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)
#     vocab_size = len(word2id_freq)
#     print("there are totoally %d different words in the corpus" % vocab_size)
#     for _, (word, word_id) in zip(range(50), word2id_dict.items()):
#         print("word %s, its id %d, its word freq %d" % (word, word_id, word2id_freq[word_id]))
#     #得到word2id词典后，我们还需要进一步处理原始语料，把每个词替换成对应的ID，便于神经网络进行处理
#     corpus = convert_corpus_to_id(corpus, word2id_dict)
#     print("word2id","%d tokens in the corpus" % len(corpus))
#     print(corpus[:50])
# #接下来，需要使用二次采样法处理原始文本。二次采样法的主要思想是降低高频词在语料中出现的频次，降低的方法是随机将高频的词抛弃，
#   # 频率越高，被抛弃的概率就越高，频率越低，被抛弃的概率就越低，这样像标点符号或冠词这样的高频词就会被抛弃，从而优化整个词表的词向量训练效果
#     corpus = subsampling(corpus, word2id_freq)
#     print("下采样","%d tokens in the corpus" % len(corpus))
#     print(corpus[:50])
#   #在完成语料数据预处理之后，需要构造训练数据。根据上面的描述，我们需要使用一个滑动窗口对语料从左到右扫描，在每个窗口内，中心词需要预测它的上下文，并形成训练数据。
# #在实际操作中，由于词表往往很大（50000，100000等），对大词表的一些矩阵运算（如softmax）需要消耗巨大的资源，因此可以通过负采样的方式模拟softmax的结果
# # 给定一个中心词和一个需要预测的上下文词，把这个上下文词作为正样本。
#   #  通过词表随机采样的方式，选择若干个负样本。
#   #  把一个大规模分类问题转化为一个2分类问题，通过这种方式优化计算速度。
#     dataset = build_data(corpus, word2id_dict, word2id_freq)
#     for _, (context_word, target_word, label) in zip(range(50), dataset):
#         print("负采样","center_word %s, target %s, label %d" % (id2word_dict[context_word],
#                                                    id2word_dict[target_word], label))
#
#   # 开始训练，定义一些训练过程中需要使用的超参数
#     batch_size = 512
#     epoch_num = 3
#     embedding_size = 200
#     step = 0
#     learning_rate = 0.001
#
#
# # 定义一个使用word-embedding计算cos的函数
#     def get_cos(query1_token, query2_token, embed):
#         W = embed
#         x = W[word2id_dict[query1_token]]
#         y = W[word2id_dict[query2_token]]
#         cos = np.dot(x, y) / np.sqrt(np.sum(y * y) * np.sum(x * x) + 1e-9)
#         flat = cos.flatten()
#         print("单词1 %s 和单词2 %s 的cos结果为 %f" % (query1_token, query2_token, cos))
#
#
# # 通过我们定义的CBOW类，来构造一个cbow模型网络
#     skip_gram_model = CBOW(vocab_size, embedding_size)
# # 构造训练这个网络的优化器
#     adam = torch.optimizer.Adam(learning_rate=learning_rate, parameters=skip_gram_model.parameters())
#
# # 使用build_batch函数，以mini-batch为单位，遍历训练数据，并训练网络
#     for center_words, target_words, label in build_batch(
#              dataset, batch_size, epoch_num):
#     # 使用paddle.to_tensor函数，将一个numpy的tensor，转换为飞桨可计算的tensor
#         center_words_var = torch.to_tensor(center_words)
#         target_words_var = torch.to_tensor(target_words)
#         label_var = torch.to_tensor(label)
#
#     # 将转换后的tensor送入飞桨中，进行一次前向计算，并得到计算结果
#         pred, loss = skip_gram_model(
#              center_words_var, target_words_var, label_var)
#
#     # 通过backward函数，让程序自动完成反向计算
#         loss.backward()
#     # 通过minimize函数，让程序根据loss，完成一步对参数的优化更新
#         adam.minimize(loss)
#     # 使用clear_gradients函数清空模型中的梯度，以便于下一个mini-batch进行更新
#         skip_gram_model.clear_gradients()
#
#     # 每经过100个mini-batch，打印一次当前的loss，看看loss是否在稳定下降
#         step += 1
#         if step % 100 == 0:
#             print("step %d, loss %.3f" % (step, loss.numpy()[0]))
#
#     # 经过10000个mini-batch，打印一次模型对eval_words中的10个词计算的同义词
#     # 这里我们使用词和词之间的向量点积作为衡量相似度的方法
#     # 我们只打印了5个最相似的词
#         if step % 2000 == 0:
#             embedding_matrix = skip_gram_model.embedding.weight.numpy()
#             np.save("./embedding", embedding_matrix)
#             get_cos("king", "queen", embedding_matrix)
#             get_cos("she", "her", embedding_matrix)
#             get_cos("topic", "theme", embedding_matrix)
#             get_cos("woman", "game", embedding_matrix)
#             get_cos("one", "name", embedding_matrix)
#
#     embedding_matrix = np.load('embedding.npy')
#     get_cos("king","queen",embedding_matrix)
#     get_cos("she","her",embedding_matrix)
#     get_cos("topic","theme",embedding_matrix)
#     get_cos("woman","game",embedding_matrix)
#     get_cos("one","name",embedding_matrix)
