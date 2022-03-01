from collections import defaultdict
from time import time
from typing import List


class Evaluation:
    class Profiler:
        def __init__(self, name: str, profiles_list: List):
            self.name = name
            self.profiles_list = profiles_list

        def __enter__(self):
            self.start = time()
            return True

        def __exit__(self, t, val, trace):
            self.profiles_list.append(dict(name=self.name, duration=time() - self.start))

    def __init__(self, horizon=None):
        self.events = defaultdict(list)
        self.profiles = defaultdict(list)
        self.episodes = []
        self.start = time()

        self.horizon = horizon

    def append(self, other):
        assert self.horizon == other.horizon
        for k in self.events.keys():
            self.events[k] += other.events[k]
        for k in self.profiles.keys():
            self.profiles[k] += other.profiles[k]
        self.episodes += other.episodes

    def add_event(self, name: str, **kwargs):
        now = time() - self.start

        self.events[name].append(dict(time=now, **kwargs))

        if name == 'episode':
            self.episodes.append([])
        
        if name != 'grasp' and name != 'grasp-failure':
            self.episodes[-1].append(dict(name=name, time=now, **kwargs))

    def profile(self, name: str):
        return self.Profiler(name, self.episodes[-1])
