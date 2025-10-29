class MockParliament:
    def __init__(self):
        self.decisions = {}
    def process(self, proposals):
        return {k: {'approved': True, 'proposer': k, 'approval_ratio': 1.0} for k in proposals}
