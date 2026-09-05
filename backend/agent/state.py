class InvestigationState:

    def __init__(self, transaction):

        self.transaction = transaction

        self.observations = []

        self.plan = []

        self.actions_taken = []

        self.evidence = {}

        self.reasons = []

        self.investigation_log = []

        self.confidence = 0

        self.risk_score = 0
        
        self.adaptation = None
       
        self.decision = None

        self.risk_level = None

        self.completed = False

    def log(self, step, thought, action=None, observation=None):

        self.investigation_log.append({
            "step": step,
            "thought": thought,
            "action": action,
            "observation": observation
        })
