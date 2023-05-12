from enum import Enum, auto

class Execution(Enum):
    SEQUENTIAL = auto()
    MULTI_THREADED = auto()
    MULTI_PROCESSING = auto()
    MULTI_PROCESSING_POOL = auto()
    
class Evaluation(Enum):
    TRAINING_TIME = auto()
    
    LEARNING_CURVE = auto()
    
    PERFORMANCE_EVALUATION = auto()
    LOCAL_PERFORMANCE_EVALUATION = auto()
    GLOBAL_PERFORMANCE_EVALUATION = auto()
    
    BEHAVIOR_EVALUATION = auto()
    LOCAL_BEHAVIOR_EVALUATION = auto()
    GLOBAL_BEHAVIOR_EVALUATION = auto()
