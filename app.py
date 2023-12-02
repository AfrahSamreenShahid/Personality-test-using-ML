import streamlit as st
from PIL import Image
import joblib
import pandas as pd
import sklearn
data = pd.read_csv("D:\\AFRAH SAMREEN S(U22PG507DTS002) mini project\\big5ds.csv")
# Load the pre-trained machine learning model
model = joblib.load("afrah.joblib")
# Define the predict function
def predict(ext1,ext2,ext3,ext4,ext5,ext6,ext7,ext8,ext9,ext10,est1,est2,est3,est4,est5,est6,est7,est8,est9,est10,agr1,agr2,agr3,agr4,agr5,agr6,agr7,agr8,agr9,agr10,csn1,csn2,csn3,csn4,csn5,csn6,csn7,csn8,csn9,csn10,opn1,opn2,opn3,opn4,opn5,opn6,opn7,opn8,opn9,opn10):
    # Make a prediction using the loaded model
        X = [[ext1,ext2,ext3,ext4,ext5,ext6,ext7,ext8,ext9,ext10,est1,est2,est3,est4,est5,est6,est7,est8,est9,est10,agr1,agr2,agr3,agr4,agr5,agr6,agr7,agr8,agr9,agr10,csn1,csn2,csn3,csn4,csn5,csn6,csn7,csn8,csn9,csn10,opn1,opn2,opn3,opn4,opn5,opn6,opn7,opn8,opn9,opn10]]
        y_pred = model.predict(X)
    # Return the prediction
        return y_pred[0]
# Define the Streamlit app
def app():
    st.title("BIG FIVE PERSONALITY TEST FORM")
    text = "The Big Five utilizes a combination of five traits to assess an individual personality: Openness, Conscientiousness, Extroversion, Agreeableness, and Neuroticism. An easy way to remember the five traits is through the acronym OCEAN. Everyone has a different amount of each of these traits; rather than saying people either possess a trait or don’t, the Big Five test notes that each person ranges on a spectrum between the two extremes."
    # Display the text
    st.write(text)
    img = "https://static.vecteezy.com/system/resources/previews/012/605/712/original/big-five-personality-traits-infographic-has-4-types-of-personality-such-as-agreeableness-openness-to-experience-neuroticism-conscientiousness-and-extraversion-visual-slide-presentation-vector.jpg"
    # Display the image
    st.image(img, width=500)
    # Define the personality trait scores
    st.write('For each of the following questions, rate how well the statement describes you on a scale of 1 to 5, where 1 is "Strongly Disagree" and 5 is "Strongly Agree".')
    # Input fields
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    email = st.text_input("E-mail")
    # Personality test questions
    st.header("Kindly Answer Honestly!")
    ext1 = st.slider("I am the life of the party", min_value=0, max_value=5, step=1)
    ext2 = st.slider("I feel little concern for others", min_value=0, max_value=5, step=1)
    ext3 = st.slider("I am always prepared", min_value=0, max_value=5, step=1)
    ext4 = st.slider("I get stressed out easily", min_value=0, max_value=5, step=1)
    ext5 = st.slider("I have a rich vocabulary", min_value=0, max_value=5, step=1)
    ext6 = st.slider("I don't talk a lot", min_value=0, max_value=5, step=1)
    ext7 = st.slider("I am interested in people", min_value=0, max_value=5, step=1)
    ext8 = st.slider("I leave my belongings around", min_value=0, max_value=5, step=1)
    ext9 = st.slider("I am relaxed most of the time", min_value=0, max_value=5, step=1)
    ext10 = st.slider("I have difficulty understanding abstract ideas", min_value=0, max_value=5, step=1)


    est1 = st.slider("I feel comfortable around people", min_value=0, max_value=5, step=1)
    est2 = st.slider("I insult people", min_value=0, max_value=5, step=1)
    est3 = st.slider("I pay attention to details", min_value=0, max_value=5, step=1)
    est4 = st.slider("I worry about things", min_value=0, max_value=5, step=1)
    est5 = st.slider("I have a vivid imagination", min_value=0, max_value=5, step=1)
    est6 = st.slider("I keep in the background", min_value=0, max_value=5, step=1)
    est7 = st.slider("I sympathize with others feelings", min_value=0, max_value=5, step=1)
    est8 = st.slider("I make a mess of things", min_value=0, max_value=5, step=1)
    est9 = st.slider("I seldom feel blue", min_value=0, max_value=5, step=1)
    est10 = st.slider("I am not interested in abstract ideas", min_value=0, max_value=5, step=1)

    agr1 = st.slider("I start conversations", min_value=0, max_value=5, step=1)
    agr2 = st.slider("I am not interested in other people's problems", min_value=0, max_value=5, step=1)
    agr3 = st.slider("I get chores done right away", min_value=0, max_value=5, step=1)
    agr4 = st.slider("I am easily disturbed", min_value=0, max_value=5, step=1)
    agr5 = st.slider("I have excellent ideas", min_value=0, max_value=5, step=1)
    agr6 = st.slider("I have little to say", min_value=0, max_value=5, step=1)
    agr7 = st.slider("I have a soft heart", min_value=0, max_value=5, step=1)
    agr8 = st.slider("I often forget to put things back in their proper place", min_value=0, max_value=5, step=1)
    agr9 = st.slider("I get upset easily", min_value=0, max_value=5, step=1)
    agr10 = st.slider("I do not have a good imagination", min_value=0, max_value=5, step=1)

    csn1 = st.slider("I talk to a lot of different people at parties", min_value=0, max_value=5, step=1)
    csn2 = st.slider("I am not really interested in others", min_value=0, max_value=5, step=1)
    csn3 = st.slider("I like order", min_value=0, max_value=5, step=1)
    csn4 = st.slider("I change my mood a lot", min_value=0, max_value=5, step=1)
    csn5 = st.slider("I am quick to understand things", min_value=0, max_value=5, step=1)
    csn6 = st.slider("I don't like to draw attention to myself", min_value=0, max_value=5, step=1)
    csn7 = st.slider("I take time out for others", min_value=0, max_value=5, step=1)
    csn8 = st.slider("I shirk my duties", min_value=0, max_value=5, step=1)
    csn9 = st.slider("I have frequent mood swings", min_value=0, max_value=5, step=1)
    csn10 = st.slider("I use difficult words", min_value=0, max_value=5, step=1)

    opn1 = st.slider("I don't mind being the centre of attention", min_value=0, max_value=5, step=1)
    opn2 = st.slider("I feel others emotions", min_value=0, max_value=5, step=1)
    opn3 = st.slider("I follow a schedule", min_value=0, max_value=5, step=1)
    opn4 = st.slider("I get irritated easily", min_value=0, max_value=5, step=1)
    opn5 = st.slider("I spend time reflecting on things", min_value=0, max_value=5, step=1)
    opn6 = st.slider("I am quiet around strangers", min_value=0, max_value=5, step=1)
    opn7 = st.slider("I make people feel at ease", min_value=0, max_value=5, step=1)
    opn8 = st.slider("I am exacting in my work", min_value=0, max_value=5, step=1)
    opn9 = st.slider("I often feel blue", min_value=0, max_value=5, step=1)
    opn10 = st.slider("I am full of ideas", min_value=0, max_value=5, step=1)
    # Make a prediction when the user clicks the "Predict" button
    if st.button('Submit'):
        # Call the predict function to make a prediction
        prediction = predict(ext1,ext2,ext3,ext4,ext5,ext6,ext7,ext8,ext9,ext10,est1,est2,est3,est4,est5,est6,est7,est8,est9,est10,agr1,agr2,agr3,agr4,agr5,agr6,agr7,agr8,agr9,agr10,csn1,csn2,csn3,csn4,csn5,csn6,csn7,csn8,csn9,csn10,opn1,opn2,opn3,opn4,opn5,opn6,opn7,opn8,opn9,opn10)    
   
     # Display the prediction to the user
        st.write('Here are some tips to improve yourself')
        if prediction == 0:
            st.write("Be yourself: Authenticity is key when it comes to having a great personality. You don't have to try to be someone else, or pretend to be someone you're not. Embrace your quirks and unique traits, and be proud of who you are.")
            st.write("Practice empathy: Empathy is the ability to understand and share the feelings of others. When you practice empathy, you become more attuned to the needs and emotions of those around you, and this can help you build stronger connections with people.")
            st.write("Be positive: People are naturally drawn to positive energy, so try to maintain a positive outlook on life. Focus on the good things in your life, and try to see the best in others.")
            st.write("Communicate effectively: Good communication is essential to building healthy relationships. Practice active listening, and try to express your thoughts and feelings clearly and respectfully.")
            st.write("Develop your interests: Pursuing your passions and interests can help you feel more fulfilled and confident. When you have a strong sense of purpose, you're more likely to feel good about yourself and radiate positive energy.")
            st.write("Be open-minded: Being open-minded means being willing to consider new ideas and perspectives. When you're open-minded, you're more likely to learn and grow, and this can help you become a more interesting and engaging person.")
            st.write("Practice self-care: Taking care of yourself is essential to having a great personality. Make sure to prioritize your physical and mental health, and take time to do things that bring you joy and relaxation.")

        elif prediction == 1:
            st.write("Practice mindfulness meditation to help reduce anxiety and stress.")
            st.write("Engage in regular exercise, as it has been shown to reduce symptoms of depression and anxiety.")
            st.write("Seek support from friends, family, or a mental health professional if you are struggling with your emotions.")
            st.write("Practice positive self-talk and challenge negative thoughts when they arise.")
            st.write("Create a daily routine and stick to it, as having structure can help reduce anxiety.")
            st.write("Read books and articles on a variety of topics to expand your knowledge and perspective.")
            st.write("Try new experiences, even if they are outside of your comfort zone.")
            st.write("Engage in creative activities, such as painting or writing, to encourage your imagination and curiosity.")
            st.write("Practice empathy and open-mindedness when interacting with others.")
            st.write("Reflect on your own beliefs and values, and be open to revising them if necessary.")

        elif prediction == 2:
            st.write("Practice mindfulness meditation to help reduce anxiety and stress.")
            st.write("Engage in regular exercise, as it has been shown to reduce symptoms of depression and anxiety.")
            st.write("Seek support from friends, family, or a mental health professional if you are struggling with your emotions.")
            st.write("Practice positive self-talk and challenge negative thoughts when they arise.")
            st.write("Create a daily routine and stick to it, as having structure can help reduce anxiety.")

        elif prediction == 3:
            st.write("Read books and articles on a variety of topics to expand your knowledge and perspective.")
            st.write("Try new experiences, even if they are outside of your comfort zone.")
            st.write("Engage in creative activities, such as painting or writing, to encourage your imagination and curiosity.")
            st.write("Practice empathy and open-mindedness when interacting with others.")
            st.write("Reflect on your own beliefs and values, and be open to revising them if necessary.")

        elif prediction == 4:
            st.write("Practice active listening")
            st.write("Try to put yourself in someone else's shoes and understand their perspective.")
            st.write("Practice kindness")
            st.write("Being patient and understanding when dealing with others")
            st.write("Expressing your needs and opinions in a clear and respectful manner")
            st.write("Maintain healthy relationships")
            st.write("Try to find common ground and work together towards a solution that benefits everyone involved.")

             
# Run the app
if __name__ == '__main__':
    app()
