import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_auc_score, roc_curve
import json 
import time
from matplotlib import pyplot as plt
# Uset the output from data_prep.ipynb file.
trainData = pd.read_csv("data/input/train_data_prepped.csv")
# test Data
trainLabels = pd.read_csv("data/input/train_results.csv")

True_test = pd.read_csv("data/input/test_data_prepped.csv")['text']

column_name = ['text','text_no_numerals','text_no_punc']
train_data = trainData[column_name[1]]
train_label = trainLabels['target']


# In the first step we will split the data in training and remaining dataset
X_train, X_test, y_train, y_test = train_test_split(train_data,train_label, train_size=0.10)

X_validate, X_test, y_validate, y_test = train_test_split(X_test,y_test, train_size=0.05)


print(f"Length of Train {X_train.shape} , Validate {X_test.shape} -- {round(X_test.shape[0]/X_train.shape[0],2)}% of train data")

# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
train_vectors = vectorizer.fit_transform(X_train)

# validate_vectors = vectorizer.transform(X_test)
validate_vectors = vectorizer.transform(X_validate)


print("shape of train",train_vectors.shape)
print("shape of valid", validate_vectors.shape)


# Perform classification with SVM, kernel=linear
#Try -1
# classifier_linear = svm.SVC(kernel='rbf',C=0.5,gamma=0.1)
#Try -2
classifier_SVM = svm.SVC(kernel='rbf',C=10,gamma=0.01,verbose=True)


t0 = time.time()
classifier_SVM.fit(train_vectors, y_train)
t1 = time.time()
prediction_linear = classifier_SVM.predict(validate_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(y_validate, prediction_linear, output_dict=True)

with open(f"{column_name}_rbf_c_10_gamma_auto.json", "w") as outfile:
    json.dump(report, outfile)


sentimentConvertor = {
    'negative' : '0',
    'neutral' : '1',
    'positive' : '2'
}


# Uncomment the follwing code for final upload into kaggle.
# test_vectors = vectorizer.transform(True_test)
# prediction_true_test = classifier_SVM.predict(test_vectors)

# output_file = open("output.txt","w")
# outfile.write("id,target\n")
# for i,P_value in enumerate(prediction_true_test):
#     outfile.write(f'{i},{sentimentConvertor[P_value]}\n')
# outfile.close()


#### ROC Curve ####

y_score1 = classifier_SVM.predict_proba(X_validate)[:,1]
print('roc_auc_score for SVM ', roc_auc_score(y_validate, y_score1))


false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_validate, y_score1)

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - SVM')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("roc_svm.png")