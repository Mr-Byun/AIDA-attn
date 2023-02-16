import numpy as np
import yaml
import argparse

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from user_util import load_data, build_model, get_threshold, evaluate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
    )
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
        
    train, val, test = load_data(config['DATA_X'], config['DATA_Y'], config['WINDOW_SIZE'], config['WINDOW_STEPSIZE'])
    
    # Initialize model
    model = build_model(
        config['WINDOW_SIZE'], config['INPUT_DIM'],
        config['SPATIAL_DIM'], config['TEMPORAL_DIM'],
        config['LSTM_DIM'], config['PREOUT_DIM']
    )
    
    model.summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config['ES_PATIENCE'], restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=int(config['ES_PATIENCE']/2), verbose=1)
    ]
    
    model.fit(
        test[0], test[1],
        validation_data=val,
        batch_size=config['BATCH_SIZE'],
        epochs=config['EPOCHS'],
        callbacks=callbacks
    )
    
    evaluate(
        np.squeeze(np.vstack(test[1])), np.squeeze(np.vstack(model.predict(test[0]))),
        get_threshold(np.squeeze(np.vstack(test[1])), np.squeeze(np.vstack(model.predict(test[0]))))
    )
    
    if config['MODEL_PATH'] is not None:
        model.save_weights(config['MODEL_PATH'])