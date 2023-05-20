import ast


def map_dict_to_str(dictionary):
    return {str(k): v for k, v in dictionary.items()}


def map_str_to_dict(dictionary):
    def try_eval(value):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value

    return {try_eval(k): v for k, v in dictionary.items()}
