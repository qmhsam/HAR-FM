import ast

def encode_to_string(data, indent=0, is_lowest=False):
    if isinstance(data, (int, float, str)):
        return repr(data)
    elif isinstance(data, list):
        # if is_lowest and all(isinstance(item, (int, float, str, tuple)) for item in data):
        #     return repr(data)
        items = [f"\n{' ' * (indent + 2)}{encode_to_string(item, indent + 2, False)}" for item in data]
        return f"\n{' ' * indent}[{',' .join(items)}\n{' ' * indent}]"
    elif isinstance(data, tuple):
        if is_lowest:
            return repr(data)
        items = [f"{encode_to_string(item, indent + 2, True)}" for item in data]
        return f"({', '.join(items)}{', ' if len(data) == 1 else ''})"
    elif isinstance(data, dict):
        if is_lowest and all(isinstance(value, (int, float, str)) for value in data.values()):
            return repr(data)
        items = [f"\n{' ' * (indent + 2)}{repr(key)}: {encode_to_string(value, indent + 2, True)}"
                 for key, value in data.items()]
        return f"{{{', '.join(items)}\n{' ' * indent}}}"
    else:
        print(data)
        raise ValueError(f"Unsupported type: {type(data)}")

def decode_from_string(string):
    # Remove unnecessary whitespace
    cleaned_string = ''.join(line.strip() for line in string.split('\n'))
    # print(cleaned_string)
    # Use ast.literal_add to convert the string safely back into a Python object
    return ast.literal_eval(cleaned_string)