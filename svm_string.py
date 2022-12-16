import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_auc_score
import json 
import time


trainData = pd.read_csv("data/input/train_data_prepped.csv")
# test Data
trainLabels = pd.read_csv("data/input/train_results.csv")

True_test = pd.read_csv("data/input/test_data_prepped.csv")['text']

column_name = 'text'
train_data = trainData[column_name]
train_label = trainLabels['target']


# In the first step we will split the data in training and remaining dataset
X_train, X_test, y_train, y_test = train_test_split(train_data,train_label, train_size=0.90)


print(f"Length of Train {X_train.shape} , Validate {X_test.shape} -- {round(X_test.shape[0]/X_train.shape[0],2)}% of train data")

# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(X_train)

validate_vectors = vectorizer.transform(X_test)

print("shape of train",train_vectors.shape)
print("shape of valid", validate_vectors.shape)


# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='rbf',C=10,gamma=0.1)
t0 = time.time()
classifier_linear.fit(train_vectors, y_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(validate_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(y_test, prediction_linear, output_dict=True)

with open(f"{column_name}_rbf_c_10_gamma_auto.json", "w") as outfile:
    json.dump(report, outfile)



sentimentConvertor = {
    'negative' : '0',
    'neutral' : '1',
    'positive' : '2'
}

test_vectors = vectorizer.transform(True_test)
prediction_true_test = classifier_linear.predict(test_vectors)

output_file = open("output.txt","w")
outfile.write("id,target\n")
for i,P_value in enumerate(prediction_true_test):
    outfile.write(f'{i},{sentimentConvertor[P_value]}\n')
outfile.close()
