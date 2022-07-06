from tensorflow.keras import models, layers
import tensorflow as tf
import pickle
def build_cnn():
    cnn = models.Sequential([
        layers.Conv2D(filters=30, kernel_size=(3,3), activation='relu', input_shape=(198,198,3)),
        layers.MaxPooling2D((10,10)),
        layers.Conv2D(filters=30, kernel_size=(3,3), activation='relu', input_shape=(198,198,3)),
        layers.MaxPooling2D((10,10)),
        #dense
        layers.Flatten(),
        layers.Dense(50, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    # compile
    cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #summary
    #print(cnn.summary())
    return cnn

cnn = build_cnn()
#make a pickle file on the worker with the initial weights
def weights_init():
    weights = cnn.get_weights()
    f_weights = open('/home/ubuntu/daskNotShared/weights.pkl', 'wb')
    pickle.dump(weights, f_weights)
    f_weights.close()
    
# once the weights have been trained. This brings them from the workers to this machine for validation
def weights_get():
    f_weights = open('/home/ubuntu/daskNotShared/weights.pkl','rb')
    weights = pickle.load(f_weights)
    f_weights.close()
    return weights