import torch
import os
from datetime import datetime

from config import Config
from util.trainingprocess_stage1 import TrainingProcessStage1
from util.trainingprocess_stage3 import TrainingProcessStage3
from util.knn import KNN

def main(config):    
    # hardware constraint for speed test
    torch.set_num_threads(1)

    os.environ['OMP_NUM_THREADS'] = '1'
    
    # initialization    
    torch.manual_seed(config.seed)
    print('Start time: ', datetime.now().strftime('%H:%M:%S'))

    
    # stage1 training
    print('Training start [Stage1]')
    model_stage1 = TrainingProcessStage1(config)
    index_dict = model_stage1.index_dict
    for epoch in range(config.epochs_stage1):
        print('Epoch:', epoch)
        model_stage1.train(epoch)
    
    print('Write embeddings')
    model_stage1.write_embeddings()
    print('Stage 1 finished: ', datetime.now().strftime('%H:%M:%S'))
    
    
    '''
    # KNN
    print('KNN')
    KNN(config, neighbors = 30, knn_rna_samples=20000)
    print('KNN finished: ', datetime.now().strftime('%H:%M:%S'))
    
    # stage3 training
    print('Training start [Stage3]')
    model_stage3 = TrainingProcessStage3(config, index_dict)

    for epoch in range(config.epochs_stage3):
       print('Epoch:', epoch)
       model_stage3.train(epoch)
        
    print('Write embeddings [Stage3]')
    model_stage3.write_embeddings()
    print('Stage 3 finished: ', datetime.now().strftime('%H:%M:%S'))
    
    # KNN
    print('Skipping stage 3 KNN') # skip KNN for stage 3 due to cell type labels being from full dataset rather than just training set
    #print('KNN stage3')
    #KNN(config, neighbors = 30, knn_rna_samples=20000)
    #print('KNN finished: ', datetime.now().strftime('%H:%M:%S'))
    '''

    return model_stage1, index_dict
    
if __name__ == "__main__":
    index_dict = main()
