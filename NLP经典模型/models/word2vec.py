import torch
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self,vocab_size,embed_dim):
        super(CBOW,self).__init__()
        self.input_embedding=nn.Embedding(vocab_size,embed_dim)
        self.output_embedding=nn.Embedding(vocab_size,embed_dim)
        nn.init.normal_(self.input_embedding.weight.data,mean=0.,std=0.01)
        nn.init.normal_(self.output_embedding.weight.data,mean=0.,std=0.01)
        self.loss_fn=nn.BCEWithLogitsLoss()
    def forward(self,inputs):
        context=inputs["contexts"]#上下文作为输入
        target=inputs["targets"]#中心词及负样本作为输出
        label=inputs["labels"]#中心词标签为1，负样本标签为0
        context_emdedding=self.input_embedding(context)#上下文的词向量初始化
        context_emdedding=context_emdedding.mean(1,keepdim=True)

        target_emdedding=self.output_embedding(target)
        embdding=context_emdedding*target_emdedding
        embdding=torch.sum(embdding,dim=2)

        loss=self.loss_fn(embdding,label.float())
        return loss


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGram, self).__init__()
        self.input_embedding = nn.Embedding(vocab_size, embed_dim)
        self.output_embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.input_embedding.weight.data, mean=0., std=0.01)
        nn.init.normal_(self.output_embedding.weight.data, mean=0., std=0.01)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs):
        center_ids = inputs["center_ids"]  # [bsz, ]
        context_ids = inputs[
            "context_ids"]  # [bsz, context_size] context_size = 2 * window_size + 2 * window_size * nagative_sampling_size 上下文单词 + 每个上下文单词对应的负例
        label = inputs["labels"]  # [bsz, context_size]

        center_embedding = self.input_embedding(center_ids)  # [bsz, embed_dim]
        center_embedding = center_embedding.unsqueeze(1)  # [bsz, 1, embed_dim]
        context_embedding = self.output_embedding(context_ids)  # [bsz, context_size, embed_dim]

        embedding = center_embedding * context_embedding  # [bsz, context_size, embed_dim]
        embedding = torch.sum(embedding, dim=2)  # [bsz, context_size]

        loss = self.loss_fn(embedding, label.float())

        return loss