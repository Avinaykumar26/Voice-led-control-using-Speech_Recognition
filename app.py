from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import speech_recognition as sr

# --------------------------
# 1. TRAIN NLP MODEL
# --------------------------
commands = [
    "turn on the light", "switch on", "light on", "please turn on the led", "power on",
    "turn off the light", "switch off", "light off", "please turn off the led", "power off"
]
labels = [1,1,1,1,1, 0,0,0,0,0]

cv = CountVectorizer()
X = cv.fit_transform(commands)
model = LogisticRegression()
model.fit(X, labels)

# --------------------------
# 2. DEFINE FUNCTION → Predict ON/OFF
# --------------------------
def predict_led_action(command):
    X_test = cv.transform([command])
    prediction = model.predict(X_test)[0]

    if prediction == 1:
        return "LED ON"
    else:
        return "LED OFF"

# --------------------------
# 3. SPEECH TO TEXT FUNCTION
# --------------------------
def take_voice_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You said:", text)
        return text.lower()
    except:
        print("Sorry, could not recognize.")
        return ""

# --------------------------
# 4. MAIN LOOP
# --------------------------
while True:
    print("\nSay something...")
    command = take_voice_command()

    if command == "":
        continue

    result = predict_led_action(command)
    print("RESULT:", result)

    if "exit" in command or "stop" in command:
        break
