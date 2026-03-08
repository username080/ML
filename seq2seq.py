import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

latent_dim = 256  # LSTM gizli durum boyutu
num_encoder_tokens = 100  # giriş sözlük boyutu
num_decoder_tokens = 100  # çıkış sözlük boyutu
max_encoder_seq_length = 10
max_decoder_seq_length = 10

encoder_input_data = np.random.rand(1000, max_encoder_seq_length, num_encoder_tokens)
decoder_input_data = np.random.rand(1000, max_decoder_seq_length, num_decoder_tokens)
decoder_target_data = np.random.rand(1000, max_decoder_seq_length, num_decoder_tokens)

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.summary()

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=10,
          validation_split=0.2)
