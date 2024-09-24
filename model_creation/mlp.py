from typing import List, Optional, Tuple

from keras.models import Model
from keras.layers import Layer, Dense

def autoencoder(input: Layer,
                enc_hidden: Optional[List[Layer]], 
                latent: Layer, 
                dec_hidden: Optional[List[Layer]]) -> Tuple[Model, Model]:
    '''
    Creates an MLP that represents an autoencoder in the form of: 
    
    Input Layers -> Hidden Encoder Layers -> Latent Layer -> Hidden Decoder Layers -> Output Dense Layer (based on input).

    Each layer (except for the decoder and encoder outputs) have the dropout_rate applied to it (if applicable).

    Returns two models as a tuple in the form of: (Autoencoder, Encoder).
    '''

    encoder = input

    if enc_hidden is not None:
        for layer in enc_hidden:
            encoder = layer(encoder)

    encoder = latent(encoder)
    
    encoder_model = Model(input, encoder)

    decoder = encoder
    if dec_hidden is not None:
        for layer in dec_hidden:
            decoder = layer(decoder)

    decoder = Dense(encoder_model.input.shape, activation="linear")(decoder)

    autoencoder = Model(input, decoder)

    return (autoencoder, encoder_model)
