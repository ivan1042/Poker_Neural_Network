from dataclasses import dataclass
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelSpec:
    estimator: Any
    param_grid: dict


def get_model_specs():
    return {
        "dummy": ModelSpec(estimator=DummyClassifier(strategy="most_frequent"),
            param_grid={}
                           ),
            "logistic": ModelSpec(estimator=LogisticRegression(max_iter=5000),
            param_grid={
                "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "model__class_weight": [None, "balanced"]
            }
                                  ),
            "decision_tree": ModelSpec(estimator=DecisionTreeClassifier(random_state=42,),
            param_grid={
                "model__max_depth": [2,3,4,5,6,None,],
                "model__min_samples_leaf": [1,3,5,10,],
                "model__class_weight": [None,"balanced",]
            }
                                       )
    }