import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split 
import os
import shutil
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.metrics import Precision, Recall, AUC
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
import tensorflow_addons as tfa



# 设定数据源目录
data_dir = r'D:\图形分类'
classes = ['dic', 'ace', 'normal','min', 'r', 'tri']  # 类别子文件夹

# 设定目标目录
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')


# 创建训练集和验证集的文件夹结构
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
    

# 分割数据集并复制文件
for cls in classes:
    # 获取每个类别文件夹中的所有文件
    src_files = [os.path.join(data_dir, cls, f) for f in os.listdir(os.path.join(data_dir, cls)) if os.path.isfile(os.path.join(data_dir, cls, f))]
    
    # 分割数据集
    train_files, temp_files = train_test_split(src_files, test_size=0.4, random_state=42)  # 80%训练，20%验证

    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    # 将文件复制到训练集目录
    for f in train_files:
        shutil.copy(f, os.path.join(train_dir, cls))

    # 将文件复制到验证集目录
    for f in val_files:
        shutil.copy(f, os.path.join(val_dir, cls))

    for f in test_files:
        shutil.copy(f, os.path.join(test_dir, cls))

img_width=150
img_height=150

#构建网络结构
#输入层
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(img_width,img_height,1)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.5))
model.add(Dense(6)) #分类个数
model.add(Activation('softmax'))
model.summary()
focal_loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.25, gamma=2.0)
                                        #optimizer='rmsprop'
model.compile(loss=focal_loss,optimizer='rmsprop',
              metrics=['accuracy', Precision(), Recall(), AUC()])

train_datagen=ImageDataGenerator(rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

min_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # 增加旋转范围
    width_shift_range=0.2,  # 增加宽度偏移
    height_shift_range=0.2,  # 增加高度偏移
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=6.4,
    class_mode='categorical',
    color_mode='grayscale',  # 加载为灰度图像
    classes=classes)

min_generator = min_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=25.6,
    class_mode='categorical',
    color_mode='grayscale',
    classes=classes)

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',
    color_mode='grayscale',
    classes=classes)

def mixed_data_generator(train_generator, min_generator, majority_weight):
    while True:
        if np.random.rand() < majority_weight:
            # 从多数类别生成器获取数据
            data, labels = next(train_generator)
        else:
            # 从少数类别生成器获取数据
            data, labels = next(min_generator)
        yield data, labels

mixed_generator = mixed_data_generator(train_generator, min_generator, majority_weight=0.1)
steps_per_epoch = min(train_generator.samples, min_generator.samples) // 32
print(steps_per_epoch)
validation_steps = validation_generator.samples // validation_generator.batch_size
print(validation_steps)
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

'''classes = list(train_generator.class_indices.keys())

# 为每个类计算权重
weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),  # 确保类别顺序与生成器中的相匹配
    y=train_generator.classes  # 使用训练生成器的类标签
)'''

#class_weights = {i: weight for i, weight in enumerate(weights)}
class_weights = {
    0: 1.0,   # 类别0的权重
    1: 1.0,   # 类别1的权重
    2: 100.0,   # 类别2的权重
    3: 100.0,   # 类别3的权重
    4: 100.0,   # 类别4的权重
    5: 100.0    # 类别5的权重
}

model.fit(
    mixed_generator,
    epochs=7,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    class_weight=class_weights,
    use_multiprocessing=False, workers=1
)
  
#检测结果 
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(validation_generator)
print("Test Loss:", test_loss)  
print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)
print("测试AUC值:", test_auc)
#保存模型
save_model(model, 'rst_model2.keras')