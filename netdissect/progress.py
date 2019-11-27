'''
Utilities for showing progress bars, controlling default verbosity, etc.
'''

# If the tqdm package is not available, then do not show progress bars;
# just connect print_progress to print.
try:
    from tqdm import tqdm, tqdm_notebook
except:
    tqdm = None

default_verbosity = False

def verbose_progress(verbose):
    '''
    Sets default verbosity level.  Set to True to see progress bars.
    '''
    global default_verbosity
    default_verbosity = verbose

def tqdm_terminal(it, *args, **kwargs):
    '''
    Some settings for tqdm that make it run better in resizable terminals.
    '''
    return tqdm(it, *args, dynamic_ncols=True, ascii=True,
            leave=(not nested_tqdm()), **kwargs)

def in_notebook():
    '''
    True if running inside a Jupyter notebook.
    '''
    # From https://stackoverflow.com/a/39662359/265298
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def nested_tqdm():
    '''
    True if there is an active tqdm progress loop on the stack.
    '''
    return hasattr(tqdm, '_instances') and len(tqdm._instances) > 0

def post_progress(**kwargs):
    '''
    When within a progress loop, post_progress(k=str) will display
    the given k=str status on the right-hand-side of the progress
    status bar.  If not within a visible progress bar, does nothing.
    '''
    if nested_tqdm():
        innermost = max(tqdm._instances, key=lambda x: x.pos)
        innermost.set_postfix(**kwargs)

def desc_progress(desc):
    '''
    When within a progress loop, desc_progress(str) changes the
    left-hand-side description of the loop toe the given description.
    '''
    if nested_tqdm():
        innermost = max(tqdm._instances, key=lambda x: x.pos)
        innermost.set_description(desc)

def print_progress(*args):
    '''
    When within a progress loop, post_progress(k=str) will display
    the given k=str status on the right-hand-side of the progress
    status bar.  If not within a visible progress bar, does nothing.
    '''
    if default_verbosity:
        printfn = print if tqdm is None else tqdm.write
        printfn(' '.join(str(s) for s in args))

def default_progress(verbose=None, iftop=False):
    '''
    Returns a progress function that can wrap iterators to print
    progress messages, if verbose is True.
   
    If verbose is False or if iftop is True and there is already
    a top-level tqdm loop being reported, then a quiet non-printing
    identity function is returned.

    verbose can also be set to a spefific progress function rather
    than True, and that function will be used.
    '''
    global default_verbosity
    if verbose is None:
        verbose = default_verbosity
    if not verbose or (iftop and nested_tqdm()) or tqdm is None:
        return lambda x, *args, **kw: x
    if verbose == True:
        return tqdm_notebook if in_notebook() else tqdm_terminal
    return verbose
