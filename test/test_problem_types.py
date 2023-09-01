import pytest
from Algometrix.problem_types import Ptype


def test_Ptype():
    assert Ptype("reg") == "reg"
    assert Ptype("class") == "class"

    with pytest.raises(TypeError):
        Ptype(1)
        Ptype(-9)
        Ptype(0)
        Ptype("Classification")
        Ptype("Regression")
        Ptype("Clustering")
        Ptype("@")
