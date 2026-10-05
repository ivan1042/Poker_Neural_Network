from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

def evaluate_model(comparison, fitted_searches, x_test, y_test):
    for k in range(0, 3):
        name = comparison.iloc[k]["model"]
        pipeline = fitted_searches[name].best_estimator_
        prediction = pipeline.predict(x_test)
        print(name)
        print(classification_report(y_test, prediction, digits=3, zero_division=0))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            prediction,
            labels=pipeline.classes_,
            normalize='true',
        )
    return