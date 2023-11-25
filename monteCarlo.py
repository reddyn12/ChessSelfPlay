

class Node:
    def __init__(self) -> None:
        pass

class MonteCarloTree:
    def __init__(self, game):
        self.game = game
        self.root = Node(None, game)
