import tensorflow as tf
from pardata import parse_data
from sklearn.metrics import roc_curve, precision_recall_curve, auc,confusion_matrix,f1_score
from tensorflow.keras.utils import plot_model
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


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    feature = parse_data()
    drug_feature = feature["drug_feature"]
    matrix = feature["matrix"]
    protein_feature = feature["protein_feature"]
    Label = feature["Label"]
    ratio1 = 0.6
    train_drug, left_drug = split_data(drug_feature, ratio1)
    train_matrix, left_matrix = split_data(matrix, ratio1)
    train_protein, left_protein = split_data(protein_feature, ratio1)
    train_label, left_label = split_data(Label, ratio1)

    ratio2 = 0.5
    val_drug, test_drug = split_data(left_drug,ratio2)
    val_matrix, test_matrix = split_data(left_matrix,ratio2)
    val_protein, test_protein = split_data(left_protein,ratio2)
    val_Label, test_Label = split_data(left_label,ratio2)



    mod = tf.keras.models.load_model('./saved_model/C1.tf')
    #plot_model(mod,to_file='AAA.png',show_dtype=True)
    mod.summary()
    prediction = mod.predict(x=[test_drug,test_matrix,test_protein])

    #####################evaluate our model
    fpr, tpr, thresholds_AUC = roc_curve(test_Label, prediction)
    with open('./output/our_result.csv','w') as outfile:
        outfile.write('y_label')
        outfile.write(',')
        outfile.write('y_pred')
        outfile.write('\n')
        for i in range(len(prediction)):
            outfile.write(str(test_Label[i]))
            outfile.write(',')
            outfile.write(str(prediction[i][0]))
            outfile.write('\n')
    AUC = auc(fpr, tpr)
    precision, recall, thresholds_AUPR = precision_recall_curve(test_Label, prediction)
    AUPR = auc(recall, precision)
    distance = []
    for i in range(len(tpr)):
        distance.append(tpr[i] - fpr[i])
    opt_AUC = thresholds_AUC[distance.index(max(distance))]

    y_pred = []
    for i in range(len(prediction)):
        if prediction[i] >= opt_AUC:
            y_pred.append(1)
        else:
            y_pred.append(0)
    confusion_matix = confusion_matrix(test_Label, y_pred)
    ACC = (confusion_matix[0][0] + confusion_matix[1][1]) / (
            confusion_matix[0][0] + confusion_matix[0][1] + confusion_matix[1][0] + confusion_matix[1][1])
    Sensi = confusion_matix[0][0] / (confusion_matix[0][0] + confusion_matix[0][1])
    Speci = confusion_matix[1][1] / (confusion_matix[1][1] + confusion_matix[1][0])
    F1 = f1_score(test_Label, y_pred)

    print('*************************************')
    a,b = count_ratio(train_label)
    print('Training samples:',len(train_label),', ratio:',a/b,a,b)
    a,b = count_ratio(val_Label)
    print('Training samples:',len(val_Label),', ratio:',a/b,a,b)
    a,b = count_ratio(test_Label)
    print('Training samples:',len(test_Label),', ratio:',a/b,a,b)
    print('*************************************')

    print('\n\n\n')
    print("\t \t  ACC:\t  ", ACC)
    print("\t \t  AUC:\t  ", AUC)
    print("\t \t AUPR:\t  ", AUPR)
    print("\t \t  Sensitivity:\t  ", Sensi)
    print("\t \t  Specificity:\t  ", Speci)
    print("\t \t  F1_score:\t  ", F1)
    print('\t optimal threshold(F1_score): \t ', opt_AUC)
    print("=================================================")