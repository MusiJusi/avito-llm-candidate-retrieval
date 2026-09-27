"""Lightweight offline prediction adapter for a fitted CatBoost ranker."""
class CatBoostContextModel:
    def __init__(self, model):
        self.model = model

    def predict(self, features, num_iteration=None):
        end = min(num_iteration or self.model.tree_count_, self.model.tree_count_)
        return self.model.predict(features, ntree_end=end, thread_count=8)
