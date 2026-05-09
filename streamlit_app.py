import streamlit as st
from deepface import DeepFace
import cv2
import json

# PAGE CONFIG
st.set_page_config(
    page_title="MoodTunes AI",
    page_icon="🎵",
    layout="centered"
)

# LOAD SONG DATA
with open("songs.json", "r", encoding="utf-8") as file:
    songs = json.load(file)

# CUSTOM CSS
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(
        135deg,
        #000000,
        #121212,
        #1DB954
    );
    color: white;
}

/* REMOVE HEADER */
header {
    visibility: hidden;
}

/* REMOVE FOOTER */
footer {
    visibility: hidden;
}

/* TITLE */
.main-title {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    color: white;
}

.green {
    color: #1DB954;
}

/* SUBTITLE */
.subtitle {
    text-align: center;
    color: #b3b3b3;
    font-size: 20px;
    margin-bottom: 30px;
}

/* INPUT */
.stTextInput input {
    background-color: #181818 !important;
    color: white !important;
    border-radius: 15px !important;
    border: 1px solid #1DB954 !important;
}

/* BUTTON */
.stButton button {
    width: 100%;
    background-color: #1DB954;
    color: white;
    border: none;
    border-radius: 30px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

/* EMOTION BOX */
.emotion-box {
    background-color: #181818;
    border: 1px solid #1DB954;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    font-size: 28px;
    margin-top: 20px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown("""
<div class="main-title">
Mood<span class="green">Tunes</span> AI 🎵
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
AI Emotion Based Music Recommendation System 💚
</div>
""", unsafe_allow_html=True)

# USER INPUT
user_feeling = st.text_input(
    "💭 How are you feeling today?"
)

# BUTTON
if st.button("🎥 Detect My Emotion"):

    with st.spinner("Analyzing Emotion..."):

        cap = cv2.VideoCapture(0)

        ret, frame = cap.read()

        if ret:

            # SHOW IMAGE
            st.image(
                frame,
                channels="BGR",
                use_container_width=True
            )

            # DETECT EMOTION
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion = result[0]['dominant_emotion']

            # SHOW EMOTION
            st.markdown(
                f"""
                <div class="emotion-box">
                    😊 Detected Emotion: {emotion.upper()}
                </div>
                """,
                unsafe_allow_html=True
            )

            # SONG SECTION
            st.subheader("🎶 Recommended Songs")

            # SHOW SONGS
            if emotion in songs:

                for song in songs[emotion]:

                    st.markdown("---")

                    st.markdown(
                        f"### 🎧 {song['title']}"
                    )

                    st.write(
                        f"🎤 Artist: {song['artist']}"
                    )

                    st.link_button(
                        "▶ Play Song",
                        song['link']
                    )

            else:
                st.warning("No songs found.")

        else:
            st.error("Webcam not detected.")

        cap.release()

# FOOTER
st.write("")
st.caption("Built with ❤️ using Python, Streamlit & DeepFace")