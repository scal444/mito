import pickle


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def string_id(lipid, geometry, zo):
    ''' Generate a string that represents a system. Used for loading and saving intermediate data '''
    return "{}_r{}_j{}_l{}_f{}_z{}".format(lipid, geometry.r_cylinder, geometry.r_junction, geometry.l_cylinder, geometry.l_flat, zo)
