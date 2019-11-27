from collections import defaultdict
from importlib import import_module

def autoimport_eval(term):
    '''
    Used to evaluate an arbitrary command-line constructor specifying
    a class, with automatic import of global module names.
    '''

    class DictNamespace(object):
        def __init__(self, d):
            self.__d__ = d
        def __getattr__(self, key):
            return self.__d__[key]

    class AutoImportDict(defaultdict):
        def __init__(self, wrapped=None, parent=None):
            super().__init__()
            self.wrapped = wrapped
            self.parent = parent
        def __missing__(self, key):
            if self.wrapped is not None:
                if key in self.wrapped:
                    return self.wrapped[key]
            if self.parent is not None:
                key = self.parent + '.' + key
            if key in __builtins__:
                return __builtins__[key]
            mdl = import_module(key)
            # Return an AutoImportDict for any namespace packages
            if hasattr(mdl, '__path__'): # and not hasattr(mdl, '__file__'):
                return DictNamespace(
                        AutoImportDict(wrapped=mdl.__dict__, parent=key))
            return mdl

    return eval(term, {}, AutoImportDict())

