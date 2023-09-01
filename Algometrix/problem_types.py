def Ptype(prob_type):
    if prob_type == "reg":
        return "reg"

    elif prob_type == "class":
        return "class"

    else:
        raise TypeError(
            "type should be defined and must be either class(for Classification) or reg(for Regression)"
        )
