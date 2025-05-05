import streamlit as st
import librosa
import numpy as np
import pitch_utils
import tempfile

st.set_page_config(page_title="VocaRise MVP", layout="centered")

st.title("🎵 VocaRise – AI-powered Vocal Pitch Feedback")
st.markdown("Upload a short singing clip (WAV/MP3) and get instant pitch analysis.")

uploaded_file = st.file_uploader("Upload your singing audio", type=["wav", "mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")

    st.write("⏳ Analyzing pitch...")

    try:
        pitches, times = pitch_utils.detect_pitch(tmp_path)

        st.success("✅ Pitch analysis complete!")

        st.line_chart({
            "Time (s)": times,
            "Pitch (Hz)": pitches
        })

        avg_pitch = np.nanmean(pitches)
        st.write(f"🎯 **Average pitch**: {avg_pitch:.2f} Hz")

        if avg_pitch < 120:
            st.info("Tip: Your pitch is in a lower vocal range. Try exercises for breath support and resonance.")
        elif avg_pitch > 250:
            st.info("Tip: Your pitch is in a higher vocal range. Consider warm-ups to relax vocal strain.")
        else:
            st.info("Tip: Balanced vocal range detected. Great work! Keep practicing with scale exercises.")

    except Exception as e:
        st.error(f"❌ Error analyzing pitch: {str(e)}")

