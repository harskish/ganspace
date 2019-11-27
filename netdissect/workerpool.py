'''
WorkerPool and WorkerBase for handling the common problems in managing
a multiprocess pool of workers that aren't done by multiprocessing.Pool,
including setup with per-process state, debugging by putting the worker
on the main thread, and correct handling of unexpected errors, and ctrl-C.

To use it,
1. Put the per-process setup and the per-task work in the
   setup() and work() methods of your own WorkerBase subclass.
2. To prepare the process pool, instantiate a WorkerPool, passing your
   subclass type as the first (worker) argument, as well as any setup keyword
   arguments.  The WorkerPool will instantiate one of your workers in each
   worker process (passing in the setup arguments in those processes).
   If debugging, the pool can have process_count=0 to force all the work
   to be done immediately on the main thread; otherwise all the work
   will be passed to other processes.
3. Whenever there is a new piece of work to distribute, call pool.add(*args).
   The arguments will be queued and passed as worker.work(*args) to the
   next available worker.
4. When all the work has been distributed, call pool.join() to wait for all
   the work to complete and to finish and terminate all the worker processes.
   When pool.join() returns, all the work will have been done.

No arrangement is made to collect the results of the work: for example,
the return value of work() is ignored.  If you need to collect the
results, use your own mechanism (filesystem, shared memory object, queue)
which can be distributed using setup arguments.
'''

from multiprocessing import Process, Queue, cpu_count
import signal
import atexit
import sys

class WorkerBase(Process):
    '''
    Subclass this class and override its work() method (and optionally,
    setup() as well) to define the units of work to be done in a process
    worker in a woker pool.
    '''
    def __init__(self, i, process_count, queue, initargs):
        if process_count > 0:
            # Make sure we ignore ctrl-C if we are not on main process.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.process_id = i
        self.process_count = process_count
        self.queue = queue
        super(WorkerBase, self).__init__()
        self.setup(**initargs)
    def run(self):
        # Do the work until None is dequeued
        while True:
            try:
                work_batch = self.queue.get()
            except (KeyboardInterrupt, SystemExit):
                print('Exiting...')
                break
            if work_batch is None:
                self.queue.put(None) # for another worker
                return
            self.work(*work_batch)
    def setup(self, **initargs):
        '''
        Override this method for any per-process initialization.
        Keywoard args are passed from WorkerPool constructor.
        '''
        pass
    def work(self, *args):
        '''
        Override this method for one-time initialization.
        Args are passed from WorkerPool.add() arguments.
        '''
        raise NotImplementedError('worker subclass needed')

class WorkerPool(object):
    '''
    Instantiate this object (passing a WorkerBase subclass type
    as its first argument) to create a worker pool.  Then call
    pool.add(*args) to queue args to distribute to worker.work(*args),
    and call pool.join() to wait for all the workers to complete.
    '''
    def __init__(self, worker=WorkerBase, process_count=None, **initargs):
        global active_pools
        if process_count is None:
            process_count = cpu_count()
        if process_count == 0:
            # zero process_count uses only main process, for debugging.
            self.queue = None
            self.processes = None
            self.worker = worker(None, 0, None, initargs)
            return
        # Ctrl-C strategy: worker processes should ignore ctrl-C.  Set
        # this up to be inherited by child processes before forking.
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        active_pools[id(self)] = self
        self.queue = Queue(maxsize=(process_count * 3))
        self.processes = None   # Initialize before trying to construct workers
        self.processes = [worker(i, process_count, self.queue, initargs)
                for i in range(process_count)]
        for p in self.processes:
            p.start()
        # The main process should handle ctrl-C.  Restore this now.
        signal.signal(signal.SIGINT, original_sigint_handler)
    def add(self, *work_batch):
        if self.queue is None:
            if hasattr(self, 'worker'):
                self.worker.work(*work_batch)
            else:
                print('WorkerPool shutting down.', file=sys.stderr)
        else:
            try:
                # The queue can block if the work is so slow it gets full.
                self.queue.put(work_batch)
            except (KeyboardInterrupt, SystemExit):
                # Handle ctrl-C if done while waiting for the queue.
                self.early_terminate()
    def join(self):
        # End the queue, and wait for all worker processes to complete nicely.
        if self.queue is not None:
            self.queue.put(None)
            for p in self.processes:
                p.join()
            self.queue = None
        # Remove myself from the set of pools that need cleanup on shutdown.
        try:
            del active_pools[id(self)]
        except:
            pass
    def early_terminate(self):
        # When shutting down unexpectedly, first end the queue.
        if self.queue is not None:
            try:
                self.queue.put_nowait(None)  # Nonblocking put throws if full.
                self.queue = None
            except:
                pass
        # But then don't wait: just forcibly terminate workers.
        if self.processes is not None:
            for p in self.processes:
                p.terminate()
            self.processes = None
        try:
            del active_pools[id(self)]
        except:
            pass
    def __del__(self):
        if self.queue is not None:
            print('ERROR: workerpool.join() not called!', file=sys.stderr)
            self.join()

# Error and ctrl-C handling: kill worker processes if the main process ends.
active_pools = {}
def early_terminate_pools():
    for _, pool in list(active_pools.items()):
        pool.early_terminate()

atexit.register(early_terminate_pools)

