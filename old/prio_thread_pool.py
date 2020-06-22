import multiprocessing
import concurrent.futures as cf
import heapq, uuid

class PrioThreadPool:

    def __init__(self, max_workers=-2, executor=cf.ThreadPoolExecutor):
        if max_workers == None or max_workers == 0:
            max_workers = -1
        if max_workers < 0:
            max_workers += multiprocessing.cpu_count() + 1
        assert(max_workers > 0)
        self._max_workers = max_workers
        self._ex = executor(max_workers=max_workers)
        self._taskid_heap = []
        self._task_dict = {}
        self._awake = False

    @staticmethod
    def _wait_raise(futs, return_when):
        dones, not_dones = cf.wait(futs, return_when=return_when)
        assert(len(dones) > 0)
        assert(len(dones) + len(not_dones) == len(futs))
        for fut in dones:
            if fut.exception() is not None:
                raise fut.exception()
        return dones, not_dones

    # def run(self, prio, fn, *args, then=None, **kwargs):
    #     assert prio + 0 == prio

    #     def _next_work():
    #         yield self._ex.submit(fn, *args, **kwargs)

    #     uid = uuid.uuid4()
    #     self._task_dict[uid] = {
    #         'prio': prio,
    #         'push_one': _next_work(),
    #         'then': then,
    #     }
    #     heapq.heappush(self._taskid_heap, (prio, uid))
    #     self._wakeup()

    def iter(self, prio, fn, *iterables, then=None, chunk=1):
        assert prio + 0 == prio

        # iterables = [iter(it) for it in iterables]
        iterables = zip(*iterables)

        def _work_chunk(chunk_params):
            retvals = []
            for params in chunk_params:
                rv = fn(*params)
                retvals.append(rv)
            return retvals

        def _next_chunk():
            chunk_params = []
            try:
                while True:
                    params = next(iterables)
                    chunk_params.append(params)
                    if len(chunk_params) == chunk:
                        break
            except StopIteration:
                pass
            return chunk_params

        def _next_work():
            while True:
                chunk_params = _next_chunk()
                if len(chunk_params) == 0:
                    raise StopIteration()
                yield self._ex.submit(_work_chunk, chunk_params)


        uid = uuid.uuid4()
        self._task_dict[uid] = {
            'prio': prio,
            'push_one': _next_work(),
            'then': then,
            'chunk': chunk,
        }
        heapq.heappush(self._taskid_heap, (prio, uid))
        self._wakeup()

    def _wakeup(self):
        if self._awake:
            return
        self._awake = True
        futs = dict()
        while len(self._task_dict) > 0:
            if len(futs) >= self._max_workers:
                dones, not_dones = self._wait_raise(list(futs.keys()), cf.FIRST_COMPLETED)
                for fut in dones:
                    then = futs[fut]['then']
                    if then != None:
                        for rv in fut.result():
                            then(rv)
                futs = {fut: futs[fut] for fut in not_dones}

            _, taskuid = heapq.nsmallest(1, self._taskid_heap)[0]
            task = self._task_dict[taskuid]
            try:
                fut = next(task['push_one'])
            except StopIteration:
                heapq.heappop(self._taskid_heap)
                del self._task_dict[taskuid]
            else:
                futs[fut] = task

        while len(futs) > 0:
            dones, not_dones = self._wait_raise(list(futs.keys()), cf.FIRST_COMPLETED)
            for fut in dones:
                then = futs[fut]['then']
                if then != None:
                    for rv in fut.result():
                        then(rv)
            futs = {fut: futs[fut] for fut in not_dones}

        self._awake = False
