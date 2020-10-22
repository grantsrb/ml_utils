
def get_table(checkpt, excl_keys={"hyps", "state_dict", "optim_dict"}):
    """
    Returns a dict with keys corresponding to each key in the checkpt
    excluding state_dicts and values as empty lists.

    checkpt: dict
        keys: str
            the keys of interest
        vals: NA
    excl_keys: set
        the keys to leave out
    """
    keys = set(checkpt.keys())-excl_keys
    table = {k:[] for k in keys}
    return table
