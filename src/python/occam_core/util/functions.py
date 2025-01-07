import re


def remove_extra_spaces(original_text):
    return re.sub(r'\s{2,}', ' ', original_text)
