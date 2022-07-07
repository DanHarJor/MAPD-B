from tensorflow.keras import models, layers
import tensorflow as tf
import pickle
path_weights='/weights/weights.pkl'
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

def get_w(cnn):
    layer_w = [lay.get_weights() for lay in cnn.layers]
    return layer_w
#changing the weights of the cnn
def set_w(cnn,weights):
    for lay,w in zip(cnn.layers, weights):
        lay.set_weights(w)
    return cnn

#make a pickle file on the worker with the initial weights
def weights_init():
    weights = get_w(cnn)
    f_weights = open(path_weights, 'wb')
    pickle.dump(weights, f_weights)
    f_weights.close()
    
# once the weights have been trained. This brings them from the workers to this machine for validation
def weights_get():
    f_weights = open(path_weights,'rb')
    weights = pickle.load(f_weights)
    f_weights.close()
    return weights
#now we have the best weights we need to write them to each worker so they can re-train from the best weights
def weights_set_best(weights_best):
    f_weights = open(path_weights,'wb')
    pickle.dump(weights_best,f_weights)
    f_weights.close()
    del f_weights, weights_best