import streamlit as st
from keras.models import load_model
import numpy as np
from keras.layers import Dense, LSTM, TimeDistributed, Embedding, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
import cv2
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50
import pyttsx3
from PIL import Image
from gtts import gTTS
import re

#engine = pyttsx3.init()
st.header('Image to Audio Conversion for Visually Impaired People')
vocab = np.load('vocab.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}


print("+"*50)
print("vocabulary loaded")


embedding_size = 128
vocab_size = len(vocab)
max_len = 40


image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))


language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))


conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('mine_model_weights.h5')

print("="*150)
print("MODEL LOADED")

resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')


#resnet = load_model('model.h5')

print("="*150)
print("RESNET MODEL LOADED")

def main():
    uploaded_file = st.file_uploader('Upload a Image', type=['jpg', 'png'])
    if uploaded_file is not None:
        with open('./Images/input.png', 'wb') as f:
            f.write(uploaded_file.getvalue())
           
    if st.button('Detect'): 
        image = cv2.imread('./Images/input.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (224,224))

        image = np.reshape(image, (1,224,224,3))

        
        
        incept = resnet.predict(image).reshape(1,2048)

        print("="*50)
        print("Predict Features")


        text_in = ['startofseq']

        final = ''

        print("="*50)
        print("GETING Captions")

        count = 0
        while tqdm(count < 20):

            count += 1

            encoded = []
            for i in text_in:
                encoded.append(vocab[i])

            padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

            sampled_index = np.argmax(model.predict([incept, padded]))

            sampled_word = inv_vocab[sampled_index]

            if sampled_word != 'endofseq':
                final = final + ' ' + sampled_word

            text_in.append(sampled_word)
          
          
        final_string = re.sub(r'[^\w\s]','', final)
        img = Image.open('./Images/input.png')
        st.image(img)
        st.warning(final_string)
        audio = gTTS(final_string, lang="en")
        audio.save('output.mp3')
        
        audio_file = open('output.mp3', 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='audio/mp3')
        
        
        
if __name__=='__main__':
    main()