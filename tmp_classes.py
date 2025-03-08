
class feature_map:
    def __init__(self, depth, height, width):
        self.depth = depth
        self.height = height
        self.width = width
    
    def get_size(self):
        return self.depth * self.height * self.width

class weights:
    def __init__(self, num_of_filters, depth, height = 1, width = 1):
        self.num_of_filters = num_of_filters
        self.depth = depth
        self.height = height
        self.width = width

    def get_size(self):
        return self.num_of_filters * self.depth * self.height * self.width
