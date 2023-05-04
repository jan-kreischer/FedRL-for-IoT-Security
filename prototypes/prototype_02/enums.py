from enum import Enum, auto

class Execution(Enum):
    SEQUENTIAL = auto()
    MULTI_THREADED = auto()
    MULTI_PROCESSING = auto()
    MULTI_PROCESSING_POOL = auto()
    
class Evaluation(Enum):
    TRAINING_TIME = auto()
    LEARNING_CURVE = auto()
    TEST_ACCURACY = auto()
    PERFORMANCE_EVALUATION = auto()
    LOCAL_AGENT_PERFORMANCE_EVALUATION = auto()
    GLOBAL_AGENT_PERFORMANCE_EVALUATION = auto()
    CONFUSION_MATRIX = auto()
    BEHAVIOR_ACTION_EVALUATION = auto()