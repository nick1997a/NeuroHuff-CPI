from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Concatenate,Convolution1D,GlobalMaxPooling1D,GlobalAveragePooling1D,MaxPooling1D,MaxPooling2D
from tensorflow.keras.layers import Input,Dense,BatchNormalization,Activation,Dropout,Embedding,SpatialDropout1D,Attention
from sklearn.metrics import precision_recall_curve,auc,roc_curve,f1_score
from sklearn.metrics import confusion_matrix
from pardata import parse_data
from sub import Subgraph
import tensorflow as tf
import numpy as np
def split_data(data,ratio):
    index_number = round(len(data)*ratio)
    data_one = data[:index_number]
    data_two = data[index_number:]
    return data_one,data_two
def count_ratio(Label):
    pos = 0
    neg = 0
    for i in range(len(Label)):
        if Label[i]==1:
            pos+=1
        else:
            neg+=1
    return pos, neg

class Net(object):
    def modelvv(self,initializer='glorot_normal',protein_strides=10,protein_layers=10,drug_layers=10,
                fc_layers=32,learning_rate=0.001,n_epoch=10,activation='relu',
                 dropout=0.2, filters=64,batch_size=16,decay=0.0,):
        # input drug
        input_drug = Input(shape=(100,))
        # input protein
        input_protein = Input(shape=(1500,5))
        #  Try Atom
        input_sub = Input(shape=(50,50))
        subgraph = Subgraph(batch=self.__batchsize)
        jj = subgraph([input_drug,input_sub,input_protein])
        hh = Dense(units=32,activation='elu')(jj)
        result = Dense(units=1,activation='sigmoid')(hh)

        # feature fusion
        model_final = Model(inputs=[input_drug,input_sub,input_protein],outputs=result)
        return model_final

    def __init__(self,protein_strides=15,protein_layers=None,drug_layers=512,fc_layers=None,
                learning_rate=0.0001,n_epoch=10,activation='relu',
                 dropout=0.2, filters=64,batch_size=16,decay=0.0,):

        self.__protein_strides = protein_strides
        self.__prot_layers = protein_layers
        self.__drugs_layer = drug_layers
        self.__fc_layers = fc_layers

        self.__learning_rate = learning_rate
        self.__epoch = n_epoch
        self.__activation = activation
        self.__dropout = dropout
        self.__filters = filters
        self.__batchsize = batch_size
        self.__decay = decay
        self.__model_t = self.modelvv(protein_strides=self.__protein_strides,protein_layers=self.__prot_layers,
                                      drug_layers=self.__drugs_layer,fc_layers=self.__fc_layers,
                                      learning_rate=self.__learning_rate,n_epoch=self.__epoch,
                                      activation=self.__activation,dropout=self.__dropout,
                                      filters=self.__filters,batch_size=self.__batchsize,decay=self.__decay)

        opt = Adam(lr=learning_rate,decay=self.__decay)
        self.__model_t.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

    def summary(self):
        self.__model_t.summary()
    def train(self,drug_feature,matrix,protein_feature,Label,batch_size=1,**kwargs):


        ratio1 = 0.6
        train_drug, left_drug = split_data(drug_feature, ratio1)
        train_matrix, left_matrix = split_data(matrix, ratio1)
        train_protein, left_protein = split_data(protein_feature, ratio1)
        train_label, left_label = split_data(Label, ratio1)

        ratio2 = 0.5
        val_drug, test_drug = split_data(left_drug, ratio2)
        val_matrix, test_matrix = split_data(left_matrix, ratio2)
        val_protein, test_protein = split_data(left_protein, ratio2)
        val_Label, test_Label = split_data(left_label, ratio2)

        baocun = 0
        AUC_list = []
        ACC_List = []
        AUPR_list = []
        F1_list = []
        for i in range(self.__epoch):
            self.__model_t.fit(x=[train_drug,train_matrix,train_protein],y=train_label,epochs=i+1,verbose=1,initial_epoch=i,batch_size=batch_size,shuffle=True)

