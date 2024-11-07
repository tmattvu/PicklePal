import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, TimeDistributed, Flatten, Dense,
    Bidirectional, LSTM, Masking, Lambda
)
import tensorflow.keras.backend as K

# Constants and Parameters

# Max sequence length
max_sequence_length = 20

# Court Array Dimensions
num_rows = 64    
num_cols = 32        
num_channels = 1      

num_shot_types = 13

# Win or lose
num_outcome_classes = 1 


# Bidirectional LSTM Model

# Input layer that takes in sequences of court arrays
input_layer = Input(shape=(max_sequence_length, num_rows, num_cols, num_channels), name='input_layer')

# Apply convolution to extract spatial features
conv_layer = TimeDistributed(
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    name='conv_layer'
)(input_layer)

# Flatten the output
flatten_layer = TimeDistributed(
    Flatten(),
    name='flatten_layer'
)(conv_layer)

# Reduce dimensionality (Might remove if not needed)
dense_layer = TimeDistributed(
    Dense(128, activation='relu'),
    name='dense_layer'
)(flatten_layer)

# Masking layer to ignore padded time steps
masking_layer = Masking(mask_value=0.0, name='masking_layer')(dense_layer)

# Capture temporal dependencies
lstm_layer = Bidirectional(
    LSTM(64, return_sequences=True),
    name='bilstm_layer'
)(masking_layer)

# Shot type prediction output
shot_output = TimeDistributed(
    Dense(num_shot_types, activation='softmax'),
    name='shot_output'
)(lstm_layer)

# Function to extract the last relevant output from the LSTM for point outcome prediction
def last_relevant_output(x):
    mask = K.not_equal(K.sum(x, axis=2), 0.0) 
    lengths = K.sum(K.cast(mask, 'int32'), axis=1)
    batch_size = K.shape(x)[0]
    indices = K.stack([K.arange(0, batch_size), lengths - 1], axis=1)
    last_outputs = K.tf.gather_nd(x, indices)
    return last_outputs

last_output = Lambda(last_relevant_output, name='last_relevant_output')(lstm_layer)

outcome_output = Dense(num_outcome_classes, activation='sigmoid', name='outcome_output')(last_output)

model = Model(
    inputs=input_layer, 
    outputs=[shot_output, outcome_output], 
    name='pickleball_bilstm_model'
)

# Custom loss function to handle masked shot labels
def masked_loss(y_true, y_pred):
    # Flatten the tensors
    y_true = K.flatten(y_true)
    y_pred = K.reshape(y_pred, (-1, num_shot_types))

    # Create a mask for non-padded labels
    mask = K.not_equal(y_true, -1)

    # Apply the mask to y_true and y_pred
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    # Compute loss
    loss = K.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)
    return K.mean(loss)

# Compile model
model.compile(
    optimizer='adam',
    loss={
        'shot_output': masked_loss,
        'outcome_output': 'binary_crossentropy'
    },
    metrics={
        'shot_output': 'accuracy',
        'outcome_output': 'accuracy'
    }
)
