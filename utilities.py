def map_dict_to_str(dictionary):
    return {str(k): v for k, v in dictionary.items()}


def map_str_to_dict(dictionary):
    return {eval(k): v for k, v in dictionary.items()}
