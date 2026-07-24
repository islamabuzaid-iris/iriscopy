import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#text to speech audio library
from RealtimeSTT import AudioToTextRecorder
#importing agent
import assist
#seting up the recorder with the model and language
if __name__ == "__main__":
    recorder = AudioToTextRecorder(spinner=False, model="tiny.en", language="en", post_speech_silence_duration=1.0)
    # set the hot words to listen for
    hot_words = ["iris", "hey iris", "wake up", "chop chop"]
    skip_hot_word_check = False
    print("System launched")
    # start the recorder and listen for hot words
    while True:
        text = recorder.text()
        print(text)
        # check for hot words
        if any(hot_word in text.lower() for hot_word in hot_words) or skip_hot_word_check:
            if text:
                print("User:" + text)
                recorder.stop()
                recorder.set_microphone(False)          # mute mic before Iris speaks
                #call iris
                response = assist.ask_question_memory(text)
                print("Iris:" + response)
                #call TTS and get time
            
                done =assist.TTS(response)
                recorder.set_microphone(True)           # unmute after Iris finishes speaking
                #skip to hot word after intiation
                skip_hot_word_check = True if "?" in response else False