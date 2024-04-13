import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split 
import os
from sklearn.metrics import confusion_matrix
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report


model = load_model('rst_model.keras')
img_width=150
img_height=150
data_dir = r'D:\图形分类'
test_dir = os.path.join(data_dir, 'test')
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    shuffle=False)  # 测试数据不需要打乱

predictions = model.predict(test_generator)
predicted_classes = predictions.argmax(axis=-1) 
true_classes = test_generator.classes
class_indices = test_generator.class_indices
print("Class Indices:", class_indices)
evaluation = model.evaluate(test_generator)
print(f'测试集损失: {evaluation[0]}')
print(f'测试集准确率: {evaluation[1]}')


cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:\n", cm)

def confusion(index,cm):
    FN = cm[index].sum() - cm[index, index]
    FP = cm[:, index].sum() - cm[index, index]
    TP = cm[index, index]
    TN = cm.sum() - (FP + FN + TP)    
    FPR = FP / float(FP + TN)
    return FPR
dic_index = class_indices['dic']
ace_index = class_indices['ace']     
fp_ace=confusion(ace_index,cm)
fp_dic=confusion(dic_index,cm)
print("dic假阳性率 (FPR):", fp_dic)
print("ace假阳性率 (FPR):", fp_ace)

