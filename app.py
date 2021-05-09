import streamlit as st
import numpy as np
import pandas as pd

st.title("streamlit")

messagez=st.text_input("input")
print("start")

if (messagez!=""):
    import pandas as pd
    import seaborn as sns
    df = pd.read_csv("spam.csv",encoding='latin-1')
    df.head()
    df.shape
    df.info()
    print(df['v2'].apply(lambda x: len(x.split(' '))).sum())
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df.head()

    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    df.head()

    spam = df[df["label"] == "spam"]
    spam.head()

    ham = df[df["label"] == "ham"]
    ham.head()


    df.label.value_counts()

    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.countplot(x='label',data=df)
    plt.xlabel('label')
    plt.title('Number of ham and spam messages');

    st.title("1")

    df['label'] = df['label'].map({'ham':0, 'spam':1})
    df.head()

    df['length'] = df.message.apply(len)
    df.head()

    plt.figure(figsize=(8, 5))
    df[df.label == 0].length.plot(bins=35, kind='hist', color='blue', label='Ham', alpha=0.6)
    plt.legend()
    plt.xlabel("Message Length");

    plt.figure(figsize=(8, 5))
    df[df.label == 1].length.plot(kind='hist', color='red', label='Spam', alpha=0.6)
    plt.legend()
    plt.xlabel("Message Length");

    df['label'] = df['label'].map({0:'ham', 1:'spam'})

    df.head()

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

    print(classification_report(y_true, y_pred))

    cmtx = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=['ham', 'spam']), 
        index=['ham', 'spam'], 
        columns=['ham', 'spam']
    )
    print(cmtx)

    from sklearn.model_selection import GridSearchCV
    parameters = {"var_smoothing":[1e-9, 1e-5, 1e-1]}
    gs_clf = GridSearchCV(
            GaussianNB(), parameters)
    gs_clf.fit(X_train.toarray(),y_train)

    gs_clf.best_params_

    y_true, y_pred = y_test, gs_clf.predict(X_test.toarray())
    accuracy_score(y_true, y_pred)

    cmtx = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=['ham', 'spam']), 
        index=['ham', 'spam'], 
        columns=['ham', 'spam']
    )
    print(cmtx)

    print(classification_report(y_true, y_pred))

    user_input = []
    input_msg = messagez
    user_input.append(input_msg)
    message = vectorizer.transform(user_input)
    message = message.toarray()
    predict = gs_clf.predict(message)
    print(predict[0])
    ans = predict[0]
    st.title(ans)

else :
    st.title("enter a messge to check")



