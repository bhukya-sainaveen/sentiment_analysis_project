# main.py

from src.data_processing import load_data, clean_data
from src.sentiment_analysis import calculate_sentiment
from src.hof_score_calculation import count
from src.feature_selection import label_encode
from src.classification_algorithms import split_data, decision_tree_classifier, svm_classifier, print_result

from data.word_list import hate, offensive, profane

# Load data
df = load_data('data/sample_dataset.tsv', '\t')

# DATA CLEANING
df = clean_data(df)

# SENTIMENT ANALYSIS
df['sentiment'] = df.text.apply(calculate_sentiment)

# HOF SCORE CALCULATION
df['HATE'] = df.text_split.apply(lambda x: count(x, hate))
df['OFFN'] = df.text_split.apply(lambda x: count(x, offensive))
df['PRFN'] = df.text_split.apply(lambda x: count(x, profane))


# FEATURE SELECTION
df = label_encode(df, 'task_2')

df_input = df.drop(['text_id','text','text_split','task_1','task_2','task_3','task_2_n'],axis='columns')

target = df.drop(['text_id','text','text_split','task_1','task_2','task_3','sentiment','HATE','OFFN','PRFN'], axis='columns')

# Split training and test data
X_train, X_test, y_train, y_test = split_data(df_input, target)

# APPLYING CLASSIFICATION ALGORITHMS
decision_tree_predictions = decision_tree_classifier(X_train, y_train, X_test)

svm_predictions = svm_classifier(X_train, y_train, X_test)

# Print classification results
print('Decision Tree Classifier Results:')
print_result(y_test, decision_tree_predictions)

print('\nSVM Classifier Results:')
print_result(y_test, svm_predictions)
