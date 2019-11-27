'''
Utility for simple distribution of work on multiple processes, by
making sure only one process is working on a job at once.
'''

import os, errno, socket, atexit, time, sys

def exit_if_job_done(directory):
    if pidfile_taken(os.path.join(directory, 'lockfile.pid'), verbose=True):
        sys.exit(0)
    if os.path.isfile(os.path.join(directory, 'done.txt')):
        with open(os.path.join(directory, 'done.txt')) as f:
            msg = f.read()
        print(msg)
        sys.exit(0)

def mark_job_done(directory):
    with open(os.path.join(directory, 'done.txt'), 'w') as f:
        f.write('Done by %d@%s %s at %s' %
                (os.getpid(), socket.gethostname(),
                 os.getenv('STY', ''),
                 time.strftime('%c')))

def pidfile_taken(path, verbose=False):
    '''
    Usage.  To grab an exclusive lock for the remaining duration of the
    current process (and exit if another process already has the lock),
    do this:

    if pidfile_taken('job_423/lockfile.pid', verbose=True):
        sys.exit(0)

    To do a batch of jobs, just run a script that does them all on
    each available machine, sharing a network filesystem.  When each
    job grabs a lock, then this will automatically distribute the
    jobs so that each one is done just once on one machine.
    '''

    # Try to create the file exclusively and write my pid into it.
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # If we cannot because there was a race, yield the conflicter.
            conflicter = 'race'
            try:
                with open(path, 'r') as lockfile:
                    conflicter = lockfile.read().strip() or 'empty'
            except:
                pass
            if verbose:
                print('%s held by %s' % (path, conflicter))
            return conflicter
        else:
            # Other problems get an exception.
            raise
    # Register to delete this file on exit.
    lockfile = os.fdopen(fd, 'r+')
    atexit.register(delete_pidfile, lockfile, path)
    # Write my pid into the open file.
    lockfile.write('%d@%s %s\n' % (os.getpid(), socket.gethostname(),
        os.getenv('STY', '')))
    lockfile.flush()
    os.fsync(lockfile)
    # Return 'None' to say there was not a conflict.
    return None

def delete_pidfile(lockfile, path):
    '''
    Runs at exit after pidfile_taken succeeds.
    '''
    if lockfile is not None:
        try:
            lockfile.close()
        except:
            pass
    try:
        os.unlink(path)
    except:
        pass
