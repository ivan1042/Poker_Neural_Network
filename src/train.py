import pandas as pd
from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from src.models import get_model_specs
from src.preprocessing import build_preprocessor


SCORING = {
    "macro_f1": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
    "accuracy": "accuracy",
    "log_loss": "neg_log_loss",
}


def compare_models(X_train, y_train):
    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=42,
    )

    summary = []
    fitted_searches = {}

    for name, spec in get_model_specs().items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", spec.estimator),
        ])

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec.param_grid,
            scoring=SCORING,
            refit="macro_f1",
            cv=cv,
            return_train_score=True,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)
        fitted_searches[name] = search

        best_index = search.best_index_

        summary.append({
            "model": name,
            "cv_macro_f1": search.cv_results_["mean_test_macro_f1"][best_index],
            "cv_macro_f1_std": search.cv_results_["std_test_macro_f1"][best_index],
            "train_macro_f1": search.cv_results_["mean_train_macro_f1"][best_index],
            "balanced_accuracy": search.cv_results_["mean_test_balanced_accuracy"][best_index],
            "best_parameters": search.best_params_,
        })

    comparison = (
        pd.DataFrame(summary)
        .sort_values("cv_macro_f1", ascending=False)
    )

    return comparison, fitted_searches