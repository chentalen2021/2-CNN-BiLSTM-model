#%%
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from sklearn.metrics import balanced_accuracy_score
import json
from sklearn.decomposition import PCA


#%% Read the IEMOCAP and EMODB feature set
path_IEMOCAP_features = "/Users/talen/Desktop/Audio_features_IEMOCAP.json"
path_EMODB_features = "/Users/talen/Desktop/Audio_features_EMODB.json"

with open(path_IEMOCAP_features, "r") as fp1:
    data1 = json.load(fp1)

with open(path_EMODB_features, "r") as fp2:
    data2 = json.load(fp2)


# Read the feature sets
def read_data(data, session_test_set, speaker_test_set):
    # Split the dataset into training, validation, and testing sets based on different speakers
    sessions = ["1", "2", "3", "4", "5"]
    speakers = ["F", "M"]

    # Create the test set, including the predictors X and the target Y
    X_test_LLDs = data[session_test_set][speaker_test_set]["LLDs"]
    X_test_specs = data[session_test_set][speaker_test_set]["Log-Mel-spectram"]
    X_test_smfcc = data[session_test_set][speaker_test_set]["smfcc"]
    Y_test = data[session_test_set][speaker_test_set]["class"]

    X_test_LLDs_ori = data[session_test_set][speaker_test_set]["LLDs_ori"]
    X_test_specs_ori = data[session_test_set][speaker_test_set]["Spectrogram_ori"]

    # Create the validation set similarly
    if speaker_test_set == "F":
        speaker_valid_set = "M"
    else:
        speaker_valid_set = "F"

        X_valid_LLDs = data[session_test_set][speaker_valid_set]["LLDs"]
        X_valid_specs = data[session_test_set][speaker_valid_set]["Log-Mel-spectram"]
        X_valid_smfcc = data[session_test_set][speaker_valid_set]["smfcc"]
        Y_valid = data[session_test_set][speaker_valid_set]["class"]

    X_valid_LLDs_ori = data[session_test_set][speaker_valid_set]["LLDs_ori"]
    X_valid_specs_ori = data[session_test_set][speaker_valid_set]["Spectrogram_ori"]

    # Create the training set similarly
    X_train_LLDs = []
    X_train_specs = []
    X_train_smfcc = []
    Y_train = []

    X_train_LLDs_ori = []
    X_train_specs_ori = []

    for session in sessions:
        if session != session_test_set:
            #             X_train_LLDs = X_train_LLDs + data[session]["F"]["LLDs"]+data[session]["M"]["LLDs"]
            #             X_train_specs = X_train_specs + data[session]["F"]["Log-Mel-spectram"]+data[session]["M"]["Log-Mel-spectram"]
            #             X_train_smfcc = X_train_smfcc + data[session]["F"]["smfcc"]+data[session]["M"]["smfcc"]

            #             Y_train = Y_train + data[session]["F"]["class"] + data[session]["M"]["class"]

            X_train_LLDs_ori = X_train_LLDs_ori + data[session]["F"]["LLDs_ori"] + data[session]["M"]["LLDs_ori"]
            X_train_specs_ori = X_train_specs_ori + data[session]["F"]["Spectrogram_ori"] + data[session]["M"][
                "Spectrogram_ori"]

    return np.array(X_train_LLDs), np.array(X_train_LLDs_ori), np.array(X_train_specs), np.array(X_train_specs_ori), \
           np.array(X_train_smfcc), np.array(Y_train), np.array(X_valid_LLDs), np.array(X_valid_LLDs_ori), \
           np.array(X_valid_specs), np.array(X_valid_specs_ori), np.array(X_valid_smfcc), np.array(Y_valid), \
           np.array(X_test_LLDs), np.array(X_test_LLDs_ori), np.array(X_test_specs), np.array(X_test_specs_ori),\
           np.array(X_test_smfcc), np.array(Y_test)

#%%Create the training, validation, and test sets for IEMOCAP dataset with regard to different speakers
# training : validation : test = 8 : 1 : 1
X_train_LLDs,X_train_LLDs_ori, X_train_specs, X_train_specs_ori, X_train_smfcc, Y_train, X_valid_LLDs,\
X_valid_LLDs_ori, X_valid_specs, X_valid_specs_ori, X_valid_smfcc, Y_valid, X_test_LLDs, X_test_LLDs_ori,\
X_test_specs, X_test_specs_ori , X_test_smfcc, Y_test = read_data(data1,"2","M")

X_train_specs = X_train_specs[...,np.newaxis]
X_train_smfcc = X_train_smfcc[...,np.newaxis]
X_valid_specs = X_valid_specs[...,np.newaxis]
X_valid_smfcc = X_valid_smfcc[...,np.newaxis]
X_test_specs = X_test_specs[...,np.newaxis]
X_test_smfcc = X_test_smfcc[...,np.newaxis]

print("Done!")

#%%Use PCA to select the useful features in LLDs of IEMOCAP dataset
pca = PCA(n_components=128)
X_train_LLDs_pca = X_train_LLDs.copy()
X_valid_LLDs_pca = X_valid_LLDs.copy()
X_test_LLDs_pca = X_test_LLDs.copy()

X_train_LLDs_pca = pca.fit_transform(X_train_LLDs_pca)
X_valid_LLDs_pca = pca.transform(X_valid_LLDs_pca)
X_test_LLDs_pca = pca.transform(X_test_LLDs_pca)

print("Done")


#%% Read the EMODB feature set
path_EMODB_features = "/Users/talen/Desktop/Audio_features_EMODB.json"

def read_data2(data, test_set_speaker):
    # Split the dataset into training, validation, and testing sets based on different speakers
    # '03', '08', '09', '10', '11', '12', '13', '14', '15', '16' are the ten speakers
    speakers = ["03", "08", "09", "10", "11", "12", "13", "14", "15", "16"]

    # Create the test set, including the predictors X and the target Y
    X_test_LLDs = data[test_set_speaker]["LLDs"]
    X_test_specs = data[test_set_speaker]["Log-Mel-spectram"]
    X_test_smfcc = data[test_set_speaker]["smfcc"]

    Y_test = data[test_set_speaker]["class"]

    # The original LLDs without trend removed for comparison
    X_test_LLDs_ori = data[test_set_speaker]["LLDs_ori"]
    X_test_specs_ori = data[test_set_speaker]["Spectrogram_ori"]

    # Create the validation set similarly
    speakers.remove(test_set_speaker)
    valid_set_speaker = speakers[-1]

    X_valid_LLDs = data[valid_set_speaker]["LLDs"]
    X_valid_specs = data[valid_set_speaker]["Log-Mel-spectram"]
    X_valid_smfcc = data[valid_set_speaker]["smfcc"]

    Y_valid = data[valid_set_speaker]["class"]

    X_valid_LLDs_ori = data[valid_set_speaker]["LLDs_ori"]
    X_valid_specs_ori = data[valid_set_speaker]["Spectrogram_ori"]

    # Create the training set similarly
    speakers.remove(valid_set_speaker)

    X_train_LLDs = []
    X_train_specs = []
    X_train_smfcc = []
    Y_train = []

    X_train_LLDs_ori = []
    X_train_specs_ori = []

    for speaker in speakers:
        X_train_LLDs = X_train_LLDs + data[speaker]["LLDs"]
        X_train_specs = X_train_specs + data[speaker]["Log-Mel-spectram"]
        X_train_smfcc = X_train_smfcc + data[speaker]["smfcc"]

        Y_train = Y_train + data[speaker]["class"]

        X_train_LLDs_ori = X_train_LLDs_ori + data[speaker]["LLDs_ori"]
        X_train_specs_ori = X_train_specs_ori + data[speaker]["Spectrogram_ori"]

    return np.array(X_train_LLDs), np.array(X_train_LLDs_ori), np.array(X_train_specs), np.array(X_train_specs_ori), \
           np.array(X_train_smfcc), np.array(Y_train), np.array(X_valid_LLDs), np.array(X_valid_LLDs_ori), np.array(
        X_valid_specs), \
           np.array(X_valid_specs_ori), np.array(X_valid_smfcc), np.array(Y_valid), \
           np.array(X_test_LLDs), np.array(X_test_LLDs_ori), np.array(X_test_specs), np.array(
        X_test_specs_ori), np.array(X_test_smfcc), np.array(Y_test)

#%%Create the training, validation, and test sets for EMODB dataset
# training : validation : test = 8 : 1 : 1

X_train_LLDs2, X_train_LLDs_ori2, X_train_specs2, X_train_specs_ori2, X_train_smfcc2, Y_train2, X_valid_LLDs2,\
X_valid_LLDs_ori2, X_valid_specs2, X_valid_specs_ori2, X_valid_smfcc2, Y_valid2, X_test_LLDs2, X_test_LLDs_ori2 ,\
X_test_specs2, X_test_specs_ori2, X_test_smfcc2, Y_test2 = read_data2(data2,"09")

X_train_specs2 = X_train_specs2[...,np.newaxis]
X_train_specs_ori2 = X_train_specs_ori2[..., np.newaxis]
X_train_smfcc2 = X_train_smfcc2[...,np.newaxis]

X_valid_specs2 = X_valid_specs2[...,np.newaxis]
X_valid_specs_ori2 = X_valid_specs_ori2[...,np.newaxis]
X_valid_smfcc2 = X_valid_smfcc2[...,np.newaxis]

X_test_specs2 = X_test_specs2[...,np.newaxis]
X_test_specs_ori2 = X_test_specs_ori2[...,np.newaxis]
X_test_smfcc2 = X_test_smfcc2[...,np.newaxis]

print("Done!")

#%%Use PCA to select the useful features in LLDs
pca2 = PCA(n_components=128)
X_train_LLDs_pca2 = X_train_LLDs2.copy()
X_valid_LLDs_pca2 = X_valid_LLDs2.copy()
X_test_LLDs_pca2 = X_test_LLDs2.copy()

X_train_LLDs_pca2 = pca2.fit_transform(X_train_LLDs_pca2)
X_valid_LLDs_pca2 = pca2.transform(X_valid_LLDs_pca2)
X_test_LLDs_pca2 = pca2.transform(X_test_LLDs_pca2)

print("Done")

#%% Build the three-channel model
#1.Build the model for HSF
    #Create input layer
        #6373 -> original HSF; 4973 -> HSF from trend-removed signal; 128 -> HSF after PCA
input_HSF = keras.Input(shape=(128), name="HSF_layer")
    #1st FC layer
features_HSF = keras.layers.Dense(units=1024, activation="sigmoid")(input_HSF)
    #2nd FC layer
features_HSF = keras.layers.Dense(units=512, activation="sigmoid")(features_HSF)
    #3nd FC layer
features_HSF = keras.layers.Dense(units=128, activation="sigmoid")(features_HSF)

features_HSF = keras.layers.Flatten()(features_HSF)
features_HSF

#%%
#Build the additive attention model for training the CRNN
#Source code cited from Keras attention tutorial (https://www.tensorflow.org/tutorials/text/nmt_with_attention)
class BahdanauAttention(layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.k1 = layers.Dense(units=units)
    self.k2 = layers.Dense(units=units)
    self.V = layers.Dense(1)  #the V-scaler with 1 unit

  def call(self, query, values):
    #Expand the query with a time dimension in the second axis
    #Then query shape -> (batch_size, time_step, hidden_state_size)
    query_with_time_dim = tf.expand_dims(input=query, axis=1)

    #score shape -> (batch_szie, time_step, 1)
    #Calculate the attention scores by Bahdanau attention formula
    # to take values, query and their weights into consideration
    score = self.V(tf.nn.tanh(self.k1(values) + self.k2(query_with_time_dim)))

    #attention weights shape -> (batch_size, time_step, 1)
    #Calculate the attention distribution/weights
    attention_weights = tf.nn.softmax(logits=score, axis=1)

    #context_vector shape after sum -> (batch_size, hidden_state_size)
    attention_outputs = attention_weights * values
        #Sum the outputs across the second dimension
    attention_outputs = tf.reduce_sum(attention_outputs, axis=1)

    return attention_outputs, attention_weights

#%%2.Build the model for log-mel-spectrogram
    #Create input layer with the variable input size
input_spec = keras.Input(shape=(376,40,1), name="spec_layer")
    #Create a CNN model
    #Create 1st convolution layer, followed by a max-pooling layer and a BN layer
kernerl_initialiser = tf.keras.initializers.TruncatedNormal(stddev=0.1)
activation = tf.keras.layers.LeakyReLU(0.01)
bias_initialiser = keras.initializers.Constant(value=0.1)

features_cnn1=keras.layers.Conv2D(filters=30, kernel_size=(3,3), strides=(1,1), activation=activation,padding="same",
                                  kernel_initializer=kernerl_initialiser, bias_initializer=bias_initialiser,
                                  dilation_rate = 2)(input_spec)
features_cnn1=keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='valid')(features_cnn1)
features_cnn1=keras.layers.BatchNormalization()(features_cnn1)
features_cnn1_l1=keras.layers.Dropout(0.3)(features_cnn1)

    #Create 2nd convolution layer, followed by a BN layer
features_cnn1=keras.layers.Conv2D(filters=30, kernel_size=(3,3), strides=(1,1), activation=activation,padding="same",
                                  kernel_initializer=kernerl_initialiser, bias_initializer=bias_initialiser)(features_cnn1_l1)
features_cnn1=keras.layers.BatchNormalization()(features_cnn1)
features_cnn1_l2=keras.layers.Dropout(0.3)(features_cnn1)

#     #Create the first residual block
# features_b1 = keras.layers.Add()([features_cnn1_l1,features_cnn1_l2])

#     #Create 3rd convolution layer, followed by a BN layer
# features_cnn1=keras.layers.Conv2D(filters=30, kernel_size=(3,3), strides=(1,1), activation=activation,padding="same",
#                                   kernel_initializer=kernerl_initialiser, bias_initializer=bias_initialiser)(features_b1)
# features_cnn1=keras.layers.BatchNormalization()(features_cnn1)
# features_cnn1_l3=keras.layers.Dropout(0.3)(features_cnn1)

#     #Create 4th convolution layer, followed by a BN layer
# features_cnn1=keras.layers.Conv2D(filters=30, kernel_size=(3,3), strides=(1,1), activation=activation,padding="same",
#                                   kernel_initializer=kernerl_initialiser, bias_initializer=bias_initialiser)(features_cnn1_l3)
# features_cnn1=keras.layers.BatchNormalization()(features_cnn1)
# features_cnn1_l4=keras.layers.Dropout(0.3)(features_cnn1)

#     #Create the second residual block
# features_b1 = keras.layers.Add()([features_cnn1_l3,features_cnn1_l4])

#     #Create 5th convolution layer, followed by a BN layer
# features_cnn1=keras.layers.Conv2D(filters=30, kernel_size=(3,3), strides=(1,1), activation=activation,padding="same",
#                                   kernel_initializer=kernerl_initialiser, bias_initializer=bias_initialiser)(features_cnn1)
# features_cnn1=keras.layers.BatchNormalization()(features_cnn1)
# features_cnn1=keras.layers.Dropout(0.3)(features_cnn1)

    #Reshape the output of cnn to add time_steps
features_cnn1 = tf.reshape(features_cnn1,(-1,150,1504))

    #Create Bi-LSTM layer to capture context info
features_spec = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=False),
                                           merge_mode="concat")(features_cnn1)
    #Create attention layer
Attention1 = BahdanauAttention(units=100)
features_spec,_ = Attention1.call(query=features_spec, values=features_cnn1)

    #Create FC layer to flatten the data into feature vector
features_spec = keras.layers.Flatten()(features_spec)
features_spec = keras.layers.Dense(units=128, activation="sigmoid")(features_spec)
features_spec = keras.layers.Dropout(0.3)(features_spec)

features_cnn1

#%%Build the model only based on the spectrogram channel and test its performacne
    #units=4 (4 emotion classes) -> IEMOCAP;  units=7 (7 emotion classes) -> EMODB
result_spec = keras.layers.Dense(units=7, activation="softmax", name="emotion_type")(features_spec)

model2 = keras.Model(inputs=input_spec, outputs=result_spec)

model2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["acc"]
)

model2.fit(
    x=X_train_specs2,
    y=Y_train2,
    validation_data=(X_valid_specs2, Y_valid2),
    epochs=100,
    batch_size=32,
)

#Calculate the UA of the CRNN for SER with Log-Mel-spectrogram
predictions_spec = np.argmax(model2.predict(X_test_specs2), axis=1)
ua = balanced_accuracy_score(y_true=Y_test2, y_pred=predictions_spec)
ua

#%% 3.Build the model for SMFCC
    #Create input layer
input_smfcc = keras.Input(shape=(94,14,1), name="smfcc_layer")
    #Create a CNN model
    #Create 1st convolution layer, followed by a max-pooling layer and a BN layer
features_cnn2=keras.layers.Conv2D(filters=30, kernel_size=(2,2),activation="relu", dilation_rate=2)(input_smfcc)
features_cnn2=keras.layers.MaxPool2D(pool_size=(2,1), strides=(2,1), padding='valid')(features_cnn2)
features_cnn2=keras.layers.BatchNormalization()(features_cnn2)
    #Create 2nd convolution layer, followed by a max-pooling layer and a BN layer
features_cnn2=keras.layers.Conv2D(filters=30, kernel_size=(2,2),activation="relu", dilation_rate=2)(features_cnn2)
features_cnn2=keras.layers.BatchNormalization()(features_cnn2)

    #Reshape the output of cnn to add time_steps
features_cnn2 = tf.reshape(features_cnn2,(-1,150,88))

    #Create Bi-LSTM layer to capture context info
features_smfcc = keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True),
                                           merge_mode="concat")(features_cnn2)
    #Create attention layer
Attention1 = BahdanauAttention(units=100)
features_smfcc,_ = Attention1.call(query=features_smfcc, values=features_cnn2)

    #Create FC layer to flatten the data into feature vector
features_smfcc = keras.layers.Flatten()(features_smfcc)
features_smfcc = keras.layers.Dense(units=128, activation="sigmoid")(features_smfcc)
features_smfcc = keras.layers.Dropout(0.3)(features_smfcc)

features_smfcc

#%%
#4. Use a hidden (FC) layer to concatenate and reduce the dimensionality of the three types of output features above
feature_vector = keras.layers.concatenate([features_HSF, features_spec, features_smfcc])

feature_vector = keras.layers.Dense(units=32, activation="sigmoid")(feature_vector)

#5. Output at softmax layer
#units=4 (4 emotion classes) -> IEMOCAP;  units=7 (7 emotion classes) -> EMODB
result = keras.layers.Dense(units=7, activation="softmax", name="emotion_type")(feature_vector)

result

#%%
#6. Aggregate the three channels into one model
model = keras.Model(
    inputs=[input_HSF, input_spec, input_smfcc],
    outputs=result,
)

#7. Compile and train the model
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["acc"]
)

model.fit(
    x={"HSF_layer": X_train_LLDs_pca2, "spec_layer": X_train_specs2, "smfcc_layer": X_train_smfcc2},
    y={"emotion_type": Y_train2},
    validation_data=([X_valid_LLDs_pca2, X_valid_specs2, X_valid_smfcc2],Y_valid2),
    epochs=29,
    batch_size=32,
)

#Calculate the UA of three-channel framework for SER
predictions = np.argmax(model.predict(x=[X_test_LLDs_pca2,X_test_specs2,X_test_smfcc2]), axis=1)
balanced_accuracy_score(y_true=Y_test2, y_pred=predictions)

#%%#Calculate the UA of two-channel framework (by orignial LLDs and spectrogram based on signal trend removed)

    #Concatenate the outputs from LLDs and spectrogram channels
feature_vector_two = keras.layers.concatenate([features_HSF, features_spec])

feature_vector_two = keras.layers.Dense(units=32, activation="sigmoid")(feature_vector_two)

    #Make classification by softmax
result_two = keras.layers.Dense(units=7, activation="softmax", name="emotion_type")(feature_vector_two)


model_two_channel = keras.Model(inputs=[input_HSF, input_spec], outputs=result_two)
model_two_channel.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["acc"]
)


model_two_channel.fit(
    x={"HSF_layer": X_train_LLDs_ori2, "spec_layer": X_train_specs2},
    y={"emotion_type": Y_train2},
    validation_data=([X_valid_LLDs_ori2, X_valid_specs2],Y_valid2),
    epochs=100,
    batch_size=32,
)

#Calculate the UA of two-channel framework from orignial features
predictions_ori = np.argmax(model_two_channel.predict(x=[X_test_LLDs_ori2,X_test_specs_ori2]), axis=1)
balanced_accuracy_score(y_true=Y_test2, y_pred=predictions_ori)

#%%Calculate the mean and standard deviation based on 10-fold cross validation

specs_EMD = [0.287, 0.3266, 0.5111, 0.3852, 0.3348, 0.2744, 0.2875, 0.3448, 0.412, 0.382]
three_channel_EMD = []

print("The mean is: ", np.mean(three_channel_EMD))
print("The sd is: ", np.std(three_channel_EMD))