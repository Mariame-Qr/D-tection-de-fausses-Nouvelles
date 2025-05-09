import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import pickle

# Configuration de la page
st.set_page_config(page_title="Fake News Detection", page_icon=":newspaper:", layout="wide")

# Initialisation de l'état de la session
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None

# Fonction pour changer de page
def navigate_to(page):
    st.session_state.page = page

# Barre latérale de navigation
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Clean/preprocessing", "Train Models", "Predict Text"], key='menu')

# Mettre à jour la page de l'état de session lorsque le menu est utilisé
if menu != st.session_state.page:
    navigate_to(menu)

# Inclure Bootstrap depuis un CDN
bootstrap_cdn = """
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
"""

# Style personnalisé pour les étapes
custom_css = """
<style>
    .step-card {
        background-color: #f7f7f7;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .step-card h5 {
        font-size: 1.2em;
        margin-bottom: 15px;
    }
    .step-card p {
        font-size: 1em;
        color: #555;
    }
    .step-card .status {
        margin-top: 10px;
        font-size: 0.9em;
        color: green;
    }
</style>
"""

# Rendu des pages en fonction de l'état de session
page = st.session_state.page

if page == "Home":
    st.markdown(bootstrap_cdn, unsafe_allow_html=True)
    st.markdown("""
    <div class="container mt-4">
        <div class="text-center">
            <h1>Fake News Detection</h1>
            <h3>Welcome to the Home Page</h3>
            <p class="lead">
                This web application allows you to predict whether a text is true or false using various supervised learning models. You can upload an Excel file containing the texts to be analyzed and choose from several models to get a prediction.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Clean/preprocessing":
    st.markdown(bootstrap_cdn, unsafe_allow_html=True)
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown("""
    <div class="container mt-4">
        <div class="text-center">
            <h1>Fake News Detection</h1>
            <h3>Clean/Preprocess your Data</h3>
            <p class="lead">
                Data preprocessing is the process of converting raw data into an understandable form. 
                It is also an important step in data mining, since we cannot work with raw data.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Widget de chargement de fichier pour les fichiers CSV uniquement
    uploaded_file = st.file_uploader("Select a CSV dataset", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Dataset Loaded Successfully:")
            st.write(df)

            df['fake'] = df['label'].apply(lambda x: 0 if x == "REAL" else 1)

            # Count the number of fake news
            fake_news_count = df['fake'].sum()
            total_news_count = df.shape[0]

            st.write(f"Number of fake news articles: {fake_news_count}")
            st.write(f"Total number of news articles: {total_news_count}")
            st.write(f"Proportion of fake news: {fake_news_count / total_news_count:.2f}")

            # Afficher le graphique avec Streamlit
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.figure(figsize=(8, 6))
            sns.countplot(x='fake', data=df, palette='viridis')
            plt.title('Distribution of Fake and Real News')
            plt.xlabel('News Type (0 = Real, 1 = Fake)')
            plt.ylabel('Count')
            st.pyplot()

            if st.button('Start Preprocessing'):
                # Step 1: Handle Missing Values
                df = df.dropna()

                # Step 2: Remove Duplicates
                df = df.drop_duplicates()

                # Step 3: Text Cleaning (NLP Specific)
                def clean_text(text):
                    text = re.sub(r'\W+', ' ', text)
                    text = text.lower()
                    return text

                df['cleaned_text'] = df['text'].apply(clean_text)

                # Step 4: Lowercasing (already done in clean_text)

                # Step 5: Tokenization
                df['tokens'] = df['cleaned_text'].apply(word_tokenize)

                # Step 6: Stop Words Removal
                stop_words = set(stopwords.words('english'))
                df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words])

                # Step 7: Stemming
                stemmer = PorterStemmer()
                df['tokens'] = df['tokens'].apply(lambda x: [stemmer.stem(word) for word in x])

                # Step 8: Vectorization (Here we'll use TF-IDF)
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(df['cleaned_text'])

                st.session_state.processed_df = df
                st.session_state.vectorizer = vectorizer

                st.write("Preprocessing completed successfully!")

                # Save the processed DataFrame to a CSV
                processed_csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Processed Data",
                    data=processed_csv,
                    file_name='processed_data.csv',
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

    st.markdown("""
       <div class="steps mt-5">
           <h4>Steps:</h4>
           <div class="row">
               <div class="col-md-3 step-card">
                   <h5>Step 1</h5>
                   <p>Handle Missing Values: Identify missing values and drop rows/columns with missing values.</p>
               </div>
               <div class="col-md-3 step-card">
                   <h5>Step 2</h5>
                   <p>Remove Duplicates: Identify and remove duplicate rows.</p>
               </div>
               <div class="col-md-3 step-card">
                   <h5>Step 3</h5>
                   <p>Text Cleaning (NLP Specific): Remove or replace unwanted characters (e.g., punctuation, numbers, special characters).</p>
               </div>
               <div class="col-md-3 step-card">
                   <h5>Step 4</h5>
                   <p>Lowercasing: Convert all text to lowercase to ensure uniformity.</p>
               </div>
           </div>
           <div class="row">
               <div class="col-md-3 step-card">
                   <h5>Step 5</h5>
                   <p>Tokenization: Split text into individual words or tokens.</p>
               </div>
               <div class="col-md-3 step-card">
                   <h5>Step 6</h5>
                   <p>Stop Words Removal: Remove common words that do not carry significant meaning.</p>
               </div>
               <div class="col-md-3 step-card">
                   <h5>Step 7</h5>
                   <p>Stemming/Lemmatization: Reduce words to their root form.</p>
               </div>
               <div class="col-md-3 step-card">
                   <h5>Step 8</h5>
                   <p>Vectorization: Convert text data into numerical format (e.g., TF-IDF, Bag of Words, Word Embeddings).</p>
               </div>
           </div>
       </div>
       """, unsafe_allow_html=True)

elif page == "Train Models":
    st.title("Train Models")

    if st.session_state.processed_df is not None:
        df = st.session_state.processed_df
        vectorizer = st.session_state.vectorizer
        X = vectorizer.transform(df['cleaned_text'])
        y = df['fake']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": MultinomialNB(),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC()
        }

        if st.button('Train Models'):
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
                disp.plot()
                plt.title(f'Confusion Matrix for {name}')
                st.pyplot()

                # Display the classification report
                report = classification_report(y_test, y_pred, target_names=['Real', 'Fake'])
                st.write(f"**Classification Report for {name}:**")
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"**Accuracy for {name}: {accuracy:.6f}**")
                st.text(report)

                # Save the model using pickle
                model_filename = f'{name.replace(" ", "_")}_model.pkl'
                with open(model_filename, 'wb') as file:
                    pickle.dump(model, file)
                st.write(f"Model saved as {model_filename}")

    else:
        st.write("Please preprocess the data first using the 'Clean/preprocessing' section.")

elif page == "Predict Text":
    st.markdown("""
          <div class="container mt-4">
              <div class="text-center">
                  <h1>Fake News Detection</h1>
              </div>
          </div>
          """, unsafe_allow_html=True)
    st.title("Predict Text")

    # Select model for prediction
    model_choice = st.selectbox("Choose a model for prediction:",
                                ["Logistic Regression", "KNN", "Naive Bayes", "Decision Tree", "SVM"])

    # Load the selected model
    model_filenames = {
        "Logistic Regression": "RL.pkl",
        "KNN": "KNN.pkl",
        "Naive Bayes": "NB.pkl",
        "Decision Tree": "DT.pkl",
        "SVM": "SVM.pkl"
    }

    vectorizer_filename = "vectorizer.pkl"

    try:
        with open(model_filenames[model_choice], 'rb') as file:
            model = pickle.load(file)
        st.write(f"{model_choice} model loaded successfully.")
    except FileNotFoundError:
        st.error(f"Model file {model_filenames[model_choice]} not found. Please train the models first.")

    # Load the vectorizer
    try:
        with open(vectorizer_filename, 'rb') as file:
            vectorizer = pickle.load(file)
        st.write("Vectorizer loaded successfully.")
    except FileNotFoundError:
        st.error(f"Vectorizer file {vectorizer_filename} not found. Please preprocess the data first.")

    # Text input for prediction
    input_text = st.text_area("Enter text for prediction:")

    if st.button("Predict"):
        if input_text:
            # Preprocess the text
            def preprocess_text(text):
                text = re.sub(r'\W+', ' ', text)
                text = text.lower()
                tokens = word_tokenize(text)
                stop_words = set(stopwords.words('english'))
                tokens = [word for word in tokens if word not in stop_words]
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(word) for word in tokens]
                return ' '.join(tokens)


            processed_text = preprocess_text(input_text)

            # Transform the text
            text_vector = vectorizer.transform([processed_text])

            # Make prediction
            prediction = model.predict(text_vector)

            # Display the result
            if prediction[0] == 1:
                st.write("Prediction: Fake News")
            else:
                st.write("Prediction: Real News")