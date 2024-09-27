class CentroidAssignmentStragety(object):
    def __init__(self, item_code_bytes, num_items, device) -> None:
        self.item_code_bytes = item_code_bytes
        self.num_items = num_items
        self.device = device

    def assign(self, train_users):
        raise NotImplementedError()
