from gtts import gTTS
import pygame
import time

# Text to be converted to speech
text = "Hello, world!"

# Language in which you want the text to be spoken
language = "en"

# Create a gTTS object
tts = gTTS(text=text, lang=language, slow=False)

# Save the converted audio to a file
audio_file = "hello.mp3"
tts.save(audio_file)

# Initialize pygame mixer
pygame.mixer.init()

# Load the audio file
pygame.mixer.music.load(audio_file)

# Play the audio file
pygame.mixer.music.play()

# Wait for the audio to finish playing
while pygame.mixer.music.get_busy():
    time.sleep(1)
