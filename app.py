import streamlit as st
import numpy as np
import pandas as pd

st.title("SMS SPAM DETECTION")

messagez=st.text_input("Enter SMS Below")
print("start")

if (messagez!=""):
    import pandas as pd
    import seaborn as sns
    df = pd.read_csv("spam.csv",encoding='latin-1')

    

    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

    spam = df[df["label"] == "spam"]
    spam.head()


    import seaborn as sns
    import matplotlib.pyplot as plt

    df['label'] = df['label'].map({'ham':0, 'spam':1})

    df['length'] = df.message.apply(len)

    df['label'] = df['label'].map({0:'ham', 1:'spam'})

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size = 0.2, random_state = 1)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)


    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix

    clf = GaussianNB()
    clf.fit(X_train.toarray(),y_train)

    y_true, y_pred = y_test, clf.predict(X_test.toarray())
    accuracy_score(y_true, y_pred)

    cmtx = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=['ham', 'spam']), 
        index=['ham', 'spam'], 
        columns=['ham', 'spam']
    )

    from sklearn.model_selection import GridSearchCV
    parameters = {"var_smoothing":[1e-9, 1e-5, 1e-1]}
    gs_clf = GridSearchCV(
            GaussianNB(), parameters)
    gs_clf.fit(X_train.toarray(),y_train)

    z = gs_clf.best_params_

    y_true, y_pred = y_test, gs_clf.predict(X_test.toarray())
    accuracy_score(y_true, y_pred)

    cmtx = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=['ham', 'spam']), 
        index=['ham', 'spam'], 
        columns=['ham', 'spam']
    )

    
    
    user_input = []
    input_msg = messagez
    user_input.append(input_msg)
    message = vectorizer.transform(user_input)
    message = message.toarray()
    predict = gs_clf.predict(message)
    ans = predict[0]
    if (ans=="spam"):
        st.title("OH NO IT'S A SPAM MESSAGE !!!")
    else :
        st.title("STAY CALM ITS A MESSAGE")

else :
    st.title("enter a messge to check")