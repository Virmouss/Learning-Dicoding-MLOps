import tensorflow as tf
import tensorflow_transform as tft 
import keras
from keras import layers
import keras_tuner
import os  
import tensorflow_hub as hub
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components.tuner.component import TunerFnResult
 
LABEL_KEY = "sentiment"
FEATURE_KEY = "review"
 
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"
 
def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
 
 
def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs,
             batch_size=64)->tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""
    
    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    
    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = transformed_name(LABEL_KEY))
    return dataset

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)

def model_builder(hp):
    inputs = keras.Input(shape=(1,), name= transformed_name(FEATURE_KEY), dtype=(tf.string))
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)

    hp_embedding_dim = hp.Int('embedding_dim', min_value=16, max_value=64, step=16)
    x = layers.Embedding(VOCAB_SIZE, hp_embedding_dim, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)

    hp_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    x = layers.Dense(hp_units, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    return model

def tuner_fn(fn_args):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files[0], tf_transform_output, num_epochs=5)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, num_epochs=5)

    # Menggunakan Hyperband terlalu lama

    # tuner = keras_tuner.Hyperband(model_builder,
    #                               objective='val_binary_accuracy',
    #                               max_epochs=10,
    #                               factor=3,
    #                               directory=fn_args.working_dir,
    #                               project_name='review_tuning')
    
    tuner = keras_tuner.RandomSearch(
        model_builder,
        objective='val_binary_accuracy',
        max_trials=3,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name='review_tuning'
    )
    
    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
                for i in list(train_set)]]
    )
    fit_kwargs = {
        "x": train_set,
        "validation_data": val_set,
        # "steps_per_epoch": 100,
        # "validation_steps": 50,
        "callbacks": [tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=3)]
    }

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs=fit_kwargs
    )