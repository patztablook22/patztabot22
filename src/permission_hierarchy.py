import pickle

class PermissionHierarchy:
    def __init__(self, config, cache_path):
        self.cache_path = cache_path
        self.levels = ['ignored', 'visible', 'mod', 'admin', 'owner']
        self.default = config['default']
        self.saved = config['saved']
        try:
            self.load()
        except:
            pass

    def save(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump((self.default, self.saved), f)

    def load(self):
        with open(self.cache_path, 'rb') as f:
            self.default, self.saved = pickle.load(f)

    def get_saved_level(self, uid):
        if uid in self.saved:
            return self.saved[uid]
        else:
            return -1

    def get_level(self, uid):
        if uid in self.saved:
            return self.saved[uid]
        else:
            return self.default

    def visible(self, uid):
        return self.get_level(uid) >= 1

    def mod(self, uid):
        return self.get_level(uid) >= 2

    def admin(self, uid):
        return self.get_level(uid) >= 3

    def owner(self, uid):
        return self.get_level(uid) >= 4

    def set(self, target_uid, new_level):
        self.saved[target_uid] = new_level
        self.save()

    def unset(self, target_uid):
        if target_uid in self.saved:
            del self.saved[target_uid]
        self.save()
