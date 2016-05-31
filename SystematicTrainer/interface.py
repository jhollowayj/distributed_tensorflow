import abc

class World:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def start(self):
        return

    @abc.abstractmethod
    def is_running(self):
        return

    @abc.abstractmethod
    def get_state(self):
        return

    @abc.abstractmethod
    def act(self, action):
        return

    @abc.abstractmethod
    def get_time(self):
        return

    @abc.abstractmethod
    def get_score(self):
        return

    @abc.abstractmethod
    def get_action_space(self):
        return

    @abc.abstractmethod
    def get_state_space(self):
        return

    @abc.abstractmethod
    def reset(self):
        return

    @abc.abstractmethod
    def load(self):
        return