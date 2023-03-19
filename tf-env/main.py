from vae import VAETrainer
from ae import AETrainer
from svdd import SVDDTrainer
from datasets import TrainLoader, TestLoader
import logging
import sys
import csv

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
import argparse

from tensorflow import keras
import tensorflow as tf
from config import Config

configuration = Config()

def train_vae(datasets, config=None, input_shape=(128, 128, 1), latent_dim=16, lr=5e-4, epochs=100, batch_size=16):
    model = VAETrainer(config, input_shape, latent_dim)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    model.fit(datasets, epochs=epochs, batch_size=batch_size, shuffle=True)
    
    return model

def train_ae(datasets, config=None, input_shape=(128, 128, 1), latent_dim=16, lr=5e-4, epochs=100, batch_size=16):
    model = AETrainer(config, input_shape, latent_dim)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    model.fit(datasets, epochs=epochs, batch_size=batch_size, shuffle=True)
    
    return model

def train_svdd(datasets, objective='one-class', R=0, nu=0.02, latent_dim=16, batch_size=16, \
               ae_lr=5e-4, ae_lr_milestones=[50], ae_epochs=100, epochs=200, lr=0.0001, lr_milestones=[100, 200]):
    model = SVDDTrainer(objective=objective, R=0,  nu=nu, latent_dim=latent_dim)
    model.pretrain(datasets, ae_epochs=ae_epochs, batch_size=batch_size, ae_lr=ae_lr, ae_lr_milestones=ae_lr_milestones)

    Batches = [datasets[k:k+batch_size] for k in range(0, len(datasets), batch_size)]
    model.train(Batches, epochs=epochs, lr=lr, lr_milestones=lr_milestones)
    
    return model

def train_shallow(datasets, modelName='oc-svm', nu=0.02, gamma=0.1):
    # Initialize model and set neural network \phi
    datasets = np.concatenate([datasets, datasets.copy()], axis=0)
    new_shape = datasets.shape[1] * datasets.shape[2] * datasets.shape[3]
    datasets = datasets.reshape((-1, new_shape))

    pca = PCA().fit(datasets)

    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (12, 6)

    fig, ax = plt.subplots()
    xi = np.arange(1, len(datasets)+1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')

    plt.xlabel('Number of Components')
    # plt.xticks(np.arange(0, 257, step=1)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(150, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

    plt.axvline(x=xi[np.min(np.where(y>0.95))], color='g', linestyle='--')
    plt.text(xi[np.min(np.where(y>0.95))] + 10, 0.25,
             f'number of components: {xi[np.min(np.where(y>0.95))]}', color = 'green', fontsize=16)

    ax.grid(axis='x')
    plt.show()

    n_components = xi[np.min(np.where(y>0.95))]
    pca = PCA(n_components=n_components)
    datasets_r = pca.fit_transform(datasets)

    if modelName=='oc-svm':
        model = OneClassSVM(nu=nu, kernel='rbf', gamma=gamma)
        model.fit(datasets_r)

    elif modelName=='isoforest':
        # Initialize model
        model = IsolationForest(max_samples=256, n_estimators=100)
        # train oc_svm
        model.fit(datasets_r)
        
        
    return model, pca
    
    


def train(modelName='vae', source='Datasets/dataset_speed_change/', batch_size=16, 
          config=None, latent_dim=16, lr=0.0001, epochs=100, objective='one-class', ae_epochs=100,
          nu=0.02, gamma=0.1, pca=None, show_roc_curve=True, show_histogram=False, seed=42, 
          Normal= ['Walking', 'Running', 'Jogging', 'TalkOnThePhone']):
    
    
    # set seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    if modelName in ['vae', 'ae', 'svdd', 'oc-svm', 'isoforest']:
        
        train_loader = TrainLoader(source=f'{source}/train/', is_spectrogram=True, shuffle=False, conjugate=True)
        test_loader = TestLoader(source=f'{source}/test')
        
    elif modelName in ['neutral-ad', 'drocc']:
        train_datasets = Dataset(source=f'{source}/train', train=True, batch_size=batch_size, num_workers=0)
        test_datasets = Dataset(source=f'{source}/test', train=False, batch_size=batch_size, num_workers=0)

        dataset = (train_datasets.load(), 
                   test_datasets.load())

        train_loader, test_loader = dataset
        
    
    if modelName=='vae':
        datasets, _, _, _ = train_loader
        input_shape = datasets.shape[1:]
        model = train_vae(datasets, config=config, input_shape=input_shape, latent_dim=latent_dim, lr=lr, epochs=epochs, batch_size=batch_size)
        
    elif modelName=='ae':
        datasets, _, _, _ = train_loader
        input_shape = datasets.shape[1:]
        model = train_ae(datasets, config=config, input_shape=input_shape, latent_dim=latent_dim, lr=lr, epochs=epochs, batch_size=batch_size)
        
    elif modelName=='svdd':
        datasets, _, _, _ = train_loader
        model = train_svdd(datasets, objective=objective, R=0, nu=nu, latent_dim=latent_dim, batch_size=batch_size, \
                           ae_lr=5e-4, ae_lr_milestones=[50], ae_epochs=ae_epochs, epochs=epochs, lr=0.0001, lr_milestones=[100, 200])
            
        
    elif modelName in ['oc-svm', 'isoforest']:
        datasets, _, _, _ = train_loader
        model, pca = train_shallow(datasets, modelName=modelName, nu=nu, gamma=gamma)
        
    else:
        print('Model does not exist !')
        sys.exit()
        
    labels, scores, test_auc = test(model, modelName, test_loader, objective=objective, pca=pca, show_roc_curve=show_roc_curve, show_histogram=show_histogram)
            
    return model, test_auc

def test(model, modelName, test_loader, objective='one-class', pca=None, show_roc_curve=True, show_histogram=False):
    
    
    xTest, labels, categories, _ = test_loader
    
    if modelName in  ['vae', 'ae']:
        encoder = model.encoder
        decoder = model.decoder
        
        try:
            mean, log_var, z = encoder.predict(xTest, verbose=0)
            reconstructions = decoder.predict(z, verbose=0)
        except:
            z = encoder.predict(xTest, verbose=0)
            reconstructions = decoder.predict(z, verbose=0)
            
        scores = tf.reduce_sum((xTest - reconstructions) ** 2, axis=(1, 2, 3))
        labels = np.array(labels, dtype=int)
        scores = np.array(scores)
        test_auc = roc_auc_score(labels, scores)
        
    if modelName=='svdd':
        scores = []
        batch_size = 16
        TestBatches = [xTest[k:k+batch_size] for k in range(0, len(xTest), batch_size)]
        for mini_batch in TestBatches:
            outputs = model.net(mini_batch)
            dist = tf.reduce_sum((outputs - model.c) ** 2, axis=1)
            if objective=='soft-boundry':
                dist = dist - model.R ** 2

            scores += dist.numpy().tolist()

        labels = np.array(labels, dtype=int)
        scores = np.array(scores)
        test_auc = roc_auc_score(labels, scores)
        
    if modelName in ['oc-svm', 'isoforest']:
        new_shape = xTest.shape[1] * xTest.shape[2] * xTest.shape[3]
        xTest = xTest.reshape((-1, new_shape))

        xTest_r = pca.transform(xTest)
        scores = model.decision_function(xTest_r)

        labels = np.array(labels, dtype=int)
        labels[labels==1] = -1
        labels[labels==0] = 1
        scores = np.array(scores)
        test_auc = roc_auc_score(labels, scores)
        
        
    if show_roc_curve:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fig = plt.figure(figsize=(8, 4))
        # plt.title(f'Receiver Operating Characteristic - VAE - TEST ID.{id}')
        plt.plot(fpr, tpr, label=f'ROC curve [AUC: {round(test_auc, 2)}]')
        plt.plot([0, 1], ls="--")
        plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend()
        plt.show() 
        
    fontdict = {'family': 'serif',
                'color':  'black',
                'weight': 'bold',
                'size': 16,
            }

    if show_histogram:
        scores_walk = scores[np.where(categories=='Walking')]
        scores_run = scores[np.where(categories=='Anomalies')]
        scores_anomaly = scores[np.where(labels==1)]
        hist = plt.figure(figsize=(8, 4))
        plt.hist(scores_walk, alpha=1, bins='auto', label='walk')
        # plt.hist(scores_run, alpha=0.8, bins=50, label='run')
        plt.hist(scores_anomaly, alpha=0.6, bins=50, label='Anomaly')
        # plt.title(f'HISTOGRRAM - ANOMALY SCORES - TEST ID.{id}', fontdict=fontdict)
        plt.xlabel('ANOMALY SCORE', fontdict=fontdict)
        plt.legend()
        plt.show()
        
    return labels, scores, test_auc


def run(seed):
    
  # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    src = configuration.source.split('/')[-2]
    log_path = f'./log/{src}'
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    log_file = log_path + f'/{configuration.modelName}.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    AUC = []

    
    # Print arguments
    logger.info('=========================================================================================')
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % configuration.source)
    logger.info('model name is %s.' % configuration.modelName)
    logger.info('batch size is %s.' % configuration.batch_size)
    logger.info('model arch is %s.' % configuration.config)
    logger.info('latent space is %s.' % configuration.latent_dim)
    logger.info('learning rate is %s.' % configuration.lr)
    logger.info('Epoch is %s.' % configuration.epochs)
    logger.info('objective is %s.' % configuration.objective)
    logger.info('AE epochs is %s' %configuration.ae_epochs)
    logger.info('nu is %s.' % configuration.nu)
    logger.info('gamme is %s.' % configuration.gamma)
    logger.info('Normal movememnts are %s' %configuration.Normal)

    
    logger.info('seed is %s' % seed)
    model, auc = train(modelName=configuration.modelName, 
                       source=configuration.source, 
                       batch_size=configuration.batch_size, 
                       config=configuration.config, 
                       latent_dim=configuration.latent_dim, 
                       lr=configuration.lr, 
                       epochs=configuration.epochs, 
                       objective=configuration.objective, 
                       ae_epochs=configuration.ae_epochs,
                       nu=configuration.nu, 
                       gamma=configuration.gamma, 
                       seed=seed, 
                       Normal=configuration.Normal)
        
        
    logger.info('test auc at seed {} is {:.2f}' .format(seed, auc))
    
    result_path = f'./results/{src}'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    if not os.path.exists(f'{result_path}/{configuration.modelName}.csv'):
        with open(f'{result_path}/{configuration.modelName}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'AUC'])
    
    with open(f'{result_path}/{configuration.modelName}.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([seed, auc])   
    
    
def parser_opt():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='seed')
    opt = parser.parse_args()
    
    return opt    
    
if __name__ == '__main__':
    
    opt = parser_opt()
    run(**vars(opt))