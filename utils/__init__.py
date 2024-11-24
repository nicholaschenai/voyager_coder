from cognitive_base.utils import tag_indent_format

def format_voyager_progs(progs):
    """
    Formats a list of Voyager programs.

    This function takes a list of dictionaries representing Voyager programs and formats them
    using the `tag_indent_format` function. Each dictionary in the list should have a 'code' key.
    If the list is empty or None, the function returns the string 'None'.

    Args:
        progs (list of dict): A list of dictionaries, where each dictionary contains a 'code' key.

    Returns:
        str: A formatted string of Voyager programs or 'None' if the input list is empty or None.
    """
    return tag_indent_format('Entry', [entry['code'] for entry in progs]) if progs else 'None'
