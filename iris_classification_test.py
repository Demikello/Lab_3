from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

assert accuracy > 0.8, f'Accuracy is too low: {accuracy}'
print('Test passed')