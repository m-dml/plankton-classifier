import yaml
import os


class Config(object):
    """
    This class wraps a dictionary so that the entries of the dictionary can be accessed by the dot-notation.
    Furthermore it is checked that no other keys than the ones in the default-config can be used.

    """
    def __init__(self, entries):
        self.new_config_class = True
        self.update(entries)

    def update(self, entries):

        if isinstance(entries, str) and os.path.isfile(entries):
            with open(os.path.abspath(entries), "r") as f:
                config_dict = yaml.safe_load(f)
            self.update(config_dict)
            return

        if not isinstance(entries, dict):
            raise TypeError(f"Entries need to be a dict or a valid config file. Got: <{entries} ({type(entries)})>")

        if self.new_config_class:
            self.new_config_class = False
            new_entries = entries
        else:
            old_config = self.__dict__
            for key, value in entries.items():
                if key in old_config.keys():
                    old_config[key] = entries[key]
                else:
                    raise KeyError(f"Trying to set a key which is not set in the default file. "
                                   f"< {key} > does not exist in default config.")

            new_entries = old_config

        for key, value in new_entries.items():
            self.__setattr__(key, value)

    def __repr__(self):
        return str(self.__dict__)
    
    def save(self, filename):
        with open(filename, "w") as outfile:
            yaml.safe_dump(self.__dict__, outfile)

