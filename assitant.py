import speech_recognition as sr
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyttsx3
import torch
import random
import sys

# Initialize the components
recognizer = sr.Recognizer()
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speed of speech
engine.setProperty('volume', 0.9)  # Set volume of speech

# Set padding token to be the same as eos_token
tokenizer.pad_token = tokenizer.eos_token  # This line adds the padding token

# Rest of your code...

faq_dict = {
    "what is your name": [
        "i'm your friendly chatbot!",
        "you can call me chatbot.",
        "i'm known as the horizon event chatbot."
    ],
    "how can i help you": [
        "i'm here to assist you with your queries.",
        "feel free to ask me anything!",
        "how may i assist you today?"
    ],
    "how are you doing": [
        "i'm just a program, but i'm functioning well!",
        "doing great, thank you for asking!",
        "i'm here and ready to help!"
    ],
    # Add more FAQs and responses as needed
}


# Function to listen and recognize speech
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)  # 5 seconds timeout
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start.")
            return None
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError:
        print("Could not request results; check your network connection.")
        return None

# Function to get response from the chatbot model
def get_response(user_input):
    # Normalize the input for matching
    normalized_input = user_input.strip().lower()
    
    if normalized_input in faq_dict:
        reply = random.choice(faq_dict[normalized_input])  # Choose a random predefined response
    else:
        inputs = tokenizer(normalized_input, return_tensors="pt", padding=True)
        reply_ids = model.generate(
            inputs['input_ids'], 
            max_length=100, 
            num_return_sequences=1, 
            do_sample=True, 
            top_p=0.95, 
            top_k=60,
            pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to eos_token_id
        )
        reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    print(f"Chatbot: {reply}")
    return reply


# Function to speak the response
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Main loop to keep the chatbot running
# Main loop to keep the chatbot running
def main():
    print("Horizon Event Chatbot is ready to chat!")
    while True:
        try:
            user_text = listen()  # Capture the user's voice input
            if user_text:
                normalized_input = user_text.strip().lower()  # Normalize user input
                if normalized_input == "goodbye":  # Check for the dead switch command
                    speak("Goodbye! Have a great day!")  # Respond before shutting down
                    sys.exit(0)
                response = get_response(user_text)  # Get a response from the chatbot model
                speak(response)  # Speak out the response to the user
        except KeyboardInterrupt:
            print("Chatbot shutting down.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

# Run the chatbot
if __name__ == "__main__":
    main()



# # Run the chatbot
# if __name__ == "__main__":
#     main()
