def load_data():
    data = []
    data_labels = []
    with open("./pos_tweets.txt", encoding='utf8') as f:
        for i in f: 
            data.append(i) 
            data_labels.append('pos')

    with open("./neg_tweets.txt", encoding='utf8') as f:
        for i in f: 
            data.append(i)
            data_labels.append('neg')

    return data, data_labels

def transform_to_features(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = False,
    )
    features = vectorizer.fit_transform(
        data
    )
    features_nd = features.toarray()
    return features_nd

def train_then_build_model(data, data_labels, features_nd):
    from sklearn.model_selection import train_test_split
    # TODO : set training % to 80%.
    X_train, X_test, y_train, y_test  = train_test_split(
        features_nd, 
        data_labels,
        train_size=0.80,
        test_size=0.20,
        random_state=1234)
    
    X_train2, X_test2, y_train2, y_test2  = train_test_split(
        data, 
        data_labels,
        train_size=0.80,
        test_size=0.20,
        random_state=1234)
    
    from sklearn.linear_model import LogisticRegression
    log_model = LogisticRegression()

    log_model = log_model.fit(X=X_train, y=y_train)
    y_pred = log_model.predict(X_test)

    # print first 10th prediction in this format:
    for i in range(10):
        print("::",y_pred[i],"::", X_test2[i])

    # print accuracy
    from sklearn.metrics import accuracy_score
    # TODO
    print("Accuracy={}".format(accuracy_score(y_test,y_pred)))

def process():
    data, data_labels = load_data()
    features_nd = transform_to_features(data)
    train_then_build_model(data, data_labels, features_nd)


process()