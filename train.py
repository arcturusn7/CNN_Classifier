# codeing: utf-8

import argparse
import logging
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.gluon.data import DataLoader
from mxnet.base import MXNetError
from mxnet import nd
import load_data
from model import CNNTextClassifier
import gluonnlp as nlp
from sklearn import metrics
import copy
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

parser = argparse.ArgumentParser(description='')
parser.add_argument('--train_file', type=str, help='File containing file representing the input TRAINING data')
parser.add_argument('--val_file', type=str, help='File containing file representing the input VALIDATION data', default=None)
parser.add_argument('--test_file', type=str, help='File containing file representing the input TEST data', default=None)
parser.add_argument('--epochs', type=int, default=10, help='Upper epoch limit')
parser.add_argument('--optimizer', type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr', type=float, help='Learning rate', default=0.00001)
parser.add_argument('--batch_size', type=int, help='Training batch size', default=16)
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)
parser.add_argument('--embedding_source', type=str, default='glove.twitter.27B.100d', help='Pre-trained embedding source name')
parser.add_argument('--fix_embedding', action='store_true', help='Fix embedding vectors instead of fine-tuning them')

args = parser.parse_args()


def train_classifier(vocabulary, data_train, data_val, data_test, ctx=mx.gpu(0)):
    """
    Main training method. Trains model and displays relevant metrics during and after training.
    """
    bt = load_data.BasicTransform(['Relevant', 'Not Relevant'])
    # set up the data loaders for each data source
    train_dataloader = mx.gluon.data.DataLoader(mx.gluon.data.SimpleDataset(data_train).transform(bt), batch_size=args.batch_size, shuffle=True)
    val_dataloader = mx.gluon.data.DataLoader(mx.gluon.data.SimpleDataset(data_val).transform(bt), batch_size=args.batch_size, shuffle=True)
    test_dataloader = mx.gluon.data.DataLoader(mx.gluon.data.SimpleDataset(data_test).transform(bt), batch_size=args.batch_size, shuffle=True)
    emb_input_dim, emb_output_dim = vocabulary.embedding.idx_to_vec.shape
    model = CNNTextClassifier(emb_input_dim, emb_output_dim, dropout=args.dropout)
    # initialize model parameters on the context ctx
    model.initialize(ctx=ctx)
    # set the embedding layer parameters to the pre-trained embedding in the vocabulary
    model.embedding.weight.set_data(vocab.embedding.idx_to_vec)
    if args.fix_embedding:
        model.embedding.collect_params().setattr('grad_req', 'null')

    trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate': args.lr})
    for epoch in range(args.epochs):
        epoch_cum_loss = 0
        for i, (data, label) in enumerate(train_dataloader):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = model(data)  # should have shape (batch_size,)
                loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
                avg_loss = loss_fn(output, label).mean()  # get the average loss over the batch
            avg_loss.backward()
            trainer.step(label.shape[0])  # update weights
            epoch_cum_loss += avg_loss.asscalar()  # needed to convert mx.nd.array value back to Python float
        val_accuracy, val_precision, val_recall, val_f1, val_avg_precision, _ = evaluate(model, val_dataloader, ctx)
        # display validation accuracies after each training epoch
        print('Epoch:', epoch)
        print('Acc:', '{:.2%}'.format(val_accuracy))
        print('Precision:', '{:.4}'.format(val_precision))
        print('Recall:', '{:.4}'.format(val_recall))
        print('Loss:', epoch_cum_loss, '\n')
    test_accuracy, test_precision, test_recall, test_f1, test_avg_precision, plot = evaluate(model, test_dataloader, ctx)
    # display test metrics after training complete
    print('Test Metrics:')
    print('Acc:', '{:.2%}'.format(test_accuracy))
    print('Precision:', '{:.4}'.format(test_precision))
    print('Recall:', '{:.4}'.format(test_recall))
    print('F1 Score:', '{:.4}'.format(test_f1))
    print('Average Precision:', '{:.4}'.format(test_avg_precision))
    plot.show()

    train_accuracy, train_precision, train_recall, train_f1, train_avg_precision, _ = evaluate(model, train_dataloader, ctx)
    # display train metrics after training complete
    print('Train Metrics:')
    print('Acc:', '{:.2%}'.format(train_accuracy))
    print('Precision:', '{:.2}'.format(train_precision))
    print('Recall:', '{:.2}'.format(train_recall))
    print('F1 Score:', '{:.2}'.format(train_f1))
    print('Average Precision:', '{:.2}'.format(train_avg_precision))


def evaluate(model, dataloader, ctx=mx.gpu(0)):
    """
    Get predictions on the dataloader items from model
    Return metrics (accuracy, precision, recall, f1, avg precision)
    """
    acc = 0
    total_correct = 0
    total = 0
    labels = []  # store the ground truth labels
    scores = []  # store the predictions/scores from the model
    for i, (data, label) in enumerate(dataloader):
        out = model(data)
        out = mx.nd.softmax(out)
        #  Model inference here:
        for j in range(out.shape[0]):
            lab = int(label[j].asscalar())
            labels.append(lab)
            # gather predictions for each item here
            scores.append(np.argmax(out[j].asnumpy()))

    acc = metrics.accuracy_score(labels, scores)
    prec = metrics.precision_score(labels, scores)
    rec = metrics.recall_score(labels, scores)
    f1 = metrics.f1_score(labels, scores)
    ap = metrics.average_precision_score(labels, scores)
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    return acc, prec, rec, f1, ap, plt


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu(0)
        _ = nd.array([0], ctx=ctx)
    except MXNetError:
        ctx = mx.cpu()
    return ctx


if __name__ == '__main__':

    # load the vocab and datasets (train, val, test)
    vocab, train_dataset, val_dataset, test_dataset = load_data.load_dataset(args.train_file, args.val_file, args.test_file, 64, args.embedding_source)
    ctx = try_gpu()
    train_classifier(vocab, train_dataset, val_dataset, test_dataset, ctx)
