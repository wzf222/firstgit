import os.path
import sys
import time
import pickle
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, AveragePooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from exploit_pred import *
from clr import OneCycleLR



def check_file_exists(file_path):
        if os.path.exists(file_path) == False:
                print("Error: provided file path '%s' does not exist!" % file_path)
                sys.exit(-1)
        return

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)

### CNN network
def cnn_architecture(input_size=1250, learning_rate=0.00001, classes=256):
# Designing input layer
    input_shape = (input_size, 1)
    img_input = Input(shape=input_shape)
    # 1st convolutional block
    x = Conv1D(2, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(
            img_input)
    x = BatchNormalization()(x)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

    x = Flatten(name='flatten')(x)

    # Classification layer
    x = Dense(2, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)

    # Logits layer
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model
    inputs = img_input
    model = Model(inputs, x, name='aes_hd_model')
    optimizer = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
# def cnn_architecture(input_size=700,learning_rate=0.00001,classes=256):
#         # 获取输入规模
#         input_shape = (input_size,1)
#         img_input = Input(shape=input_shape)
#
#         #################################################
#         # 可以修改卷积层内的参数，比如大小，激活函数等，去改变网络结构
#         # conv1D函数建立一维卷积，kernel_initializer参数定义权值初始化的方法，activation定义激活函数，padding定义填充方法，name定义第一卷积块名字
#         # 此处filters=2（filters即卷积中的卷积核数），kernel_size=1（kernel_size指定1D卷积窗口的长度.）   即有两个卷积核，每个卷积he只有一个数
#         # BatchNormalization随着网络的深度增加，每层特征值分布会逐渐的向激活函数的输出区间的上下两端（激活函数饱和区间）靠近，这样继续下去就会导致梯度消失。BN就是通过方法将该层特征值分布重新拉回标准正态分布，特征值将落在激活函数对于输入较为敏感的区间，输入的小变化可导致损失函数较大的变化，使得梯度变大，避免梯度消失，同时也可加快收敛。
#         # AveragePooling1D对一维的输入作平均池化,pool_size:表示池化窗口的大小,池化后输出为2个一维数组
#         #################################################
#
#         # 1st convolutional block
#         x = Conv1D(8, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
#         x = BatchNormalization()(x)
#         x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
#
#         # 2nd convolutional block
#         x = Conv1D(16, 50, kernel_initializer='he_uniform', activation='selu', padding='same', name='block2_conv1')(x)
#         x = BatchNormalization()(x)
#         x = AveragePooling1D(50, strides=50, name='block2_pool')(x)
#
#         # 3rd convolutional block
#         x = Conv1D(32, 3, kernel_initializer='he_uniform', activation='selu', padding='same', name='block3_conv1')(x)
#         x = BatchNormalization()(x)
#         x = AveragePooling1D(7, strides=7, name='block3_pool')(x)
#
#         x = Flatten(name='flatten')(x)
#
#         # Classification layer
#         x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
#         x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
#
#         # Logits layer
#         x = Dense(classes, activation='softmax', name='predictions')(x)
#
#         # Create model
#         inputs = img_input
#         model = Model(inputs, x, name='aes_rd')
#         optimizer = Adam(lr=learning_rate)
#         model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
#         return model


#### Training model
def train_model(X_profiling, Y_profiling, X_test, Y_test, model, save_file_name, epochs=150, batch_size=100, max_lr=1e-3):
    check_file_exists(os.path.dirname(save_file_name))

    # Save model every epoch
    save_model = ModelCheckpoint(save_file_name)

    # Get the input layer shape
    input_layer_shape = model.get_layer(index=0).input_shape

    # Sanity check
    # if input_layer_shape[1] != len(X_profiling[0]):
     #   print("Error: model input shape %d instead of %d is not expected ..." % (input_layer_shape[1], len(X_profiling[0])))
      #  sys.exit(-1)
    Reshaped_X_profiling, Reshaped_X_test  = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1)),X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # One Cycle Policy
    #lr_manager = OneCycleLR(max_lr=max_lr, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None, minimum_momentum=None,verbose=True)

    #callbacks=[save_model, lr_manager]
    callbacks=[save_model]
    history = model.fit(x=Reshaped_X_profiling, y=to_categorical(Y_profiling, num_classes=256), validation_data=(Reshaped_X_test, to_categorical(Y_test, num_classes=256)), batch_size=batch_size, verbose = 1, epochs=epochs, callbacks=callbacks)
    return history


#################################################
#################################################

#####            Initialization            ######

#################################################
#################################################

# Our folders
root = "./"
AESRD_data_folder = root+"AES_RD_dataset/"
AESRD_trained_models_folder = root+"AESRD_trained_models/"
history_folder = root+"training_history/"
predictions_folder = root+"model_predictions/"

#################################################
# 参数 百度一下就可以，神经网络常见的参数
#################################################
nb_epochs =100
batch_size = 100
input_size = 64
learning_rate = 10e-3
nb_traces_attacks = 15
nb_attacks = 100
real_key = np.load(AESRD_data_folder + "key.npy")

start = time.time()

#################################################
# data和label的载入
#################################################
(X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = (np.load(AESRD_data_folder + 'data_track_1000_10.npy'), np.load(AESRD_data_folder + 'label.npy')), (np.load(AESRD_data_folder + 'data_test_attack.npy'), np.load(AESRD_data_folder + 'attack_test_label.npy')), (np.load(AESRD_data_folder + 'profiling_plaintext_AES_RD.npy'), np.load(AESRD_data_folder + 'attack_plaintext_AES_RD.npy'))
print(X_profiling)
print(Y_profiling)

# Shuffle data
(X_profiling, Y_profiling) = shuffle_data(X_profiling, Y_profiling)

X_profiling = X_profiling.astype('float32')
X_attack = X_attack.astype('float32')

X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))


#################################################
#################################################

####                Training               ######

#################################################
#################################################

# Choose your model
model = cnn_architecture(input_size=input_size, learning_rate=learning_rate)
model_name="AES_RD"

print('\n Model name = '+model_name)


print("\n############### Starting Training #################\n")

# Record the metrics
history = train_model(X_profiling[:20000], Y_profiling[:20000], X_profiling[20000:], Y_profiling[20000:], model, AESRD_trained_models_folder + model_name, epochs=nb_epochs, batch_size=batch_size, max_lr=learning_rate)


end=time.time()
print('Temps execution = %d'%(end-start))

print("\n############### Training Done #################\n")

# Save the metrics
with open(history_folder + 'history_' + model_name, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


#################################################
#################################################

####               Prediction              ######

#################################################
#################################################

print("\n############### Starting Predictions #################\n")

predictions = model.predict(X_attack)

print("\n############### Predictions Done #################\n")

np.save(predictions_folder + 'predictions_' + model_name +'.npy', predictions)


#################################################
#################################################

####            Perform attacks            ######

#################################################
#################################################

print("\n############### Starting Attack on Test Set #################\n")

avg_rank = perform_attacks(nb_traces_attacks, predictions, nb_attacks, plt=plt_attack, key=real_key, byte=0, filename=model_name)

print("\n t_GE = ")
print(np.where(avg_rank<=0))

print("\n############### Attack on Test Set Done #################\n")
