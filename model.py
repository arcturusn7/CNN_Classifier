# codeing: utf-8

import mxnet as mx
import numpy as np
from mxnet.gluon import HybridBlock
from mxnet import gluon


class CNNTextClassifier(HybridBlock):

    def __init__(self, emb_input_dim, emb_output_dim, dropout, num_classes=2, prefix=None, params=None):
        super(CNNTextClassifier, self).__init__(prefix=prefix, params=params)

        with self.name_scope():
            self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
            self.encoder = gluon.nn.Conv2D(100, 3, activation='relu')
            self.pooler = gluon.nn.AvgPool2D()
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dropout(dropout))
                self.output.add(gluon.nn.Dense(100))
                self.output.add(gluon.nn.Dense(num_classes))

    def hybrid_forward(self, F, data):
        batch_size, tweet_size = data.shape
        embedded = self.embedding(data)
        embed_size = embedded.shape[-1]
        embedded = F.reshape(embedded, (-1, 1, tweet_size, embed_size))
        encoded = self.encoder(embedded)
        pooled = self.pooler(encoded)
        return self.output(pooled)
