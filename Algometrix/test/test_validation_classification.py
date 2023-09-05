import pytest
from Algometrix.validation_classification import validation

def test_validation():
    assert validation("all") == "OK"
    assert validation(["SVC"])
    assert validation(['AdaBoostClassifier','BaggingClassifier','BernoulliNB','CalibratedClassifierCV','GaussianNB',]) == "OK"

    with pytest.raises(TypeError):
        validation("")
        validation("SVC")
        validation("Linear")
        validation(["svc"])
        validation(['AdaBoostClassifier','BaggingClassifier','BernoulliNB','CalibratedClassifierCV','GaussianNB',"sdc"])