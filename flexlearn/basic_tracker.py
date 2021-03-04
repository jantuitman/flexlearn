import json
from dataclasses import *
from typing import Dict, List,Tuple


def grow(array, new_index, new_value):
    """Makes arrays bigger until new_index is guaranteed to fit in it."""
    while new_index > len(array) - 1:
      array.append(new_value)
    return array

# this class provides a very basic statehandler, mainly for testing purposes.
# it stores the state of the manager in a dictionary.
@dataclass
class BasicTracker:
  current_epoch: int = 0 # all epochs before current epochs are supposed to be tracked.
  max_epochs : int = 25
  learning_loss : List[List[float]] = field(default_factory=list)
  validation_loss : List[float] = field(default_factory=list)
  validation_accuracy : List[float] = field(default_factory=list)

  
  @classmethod
  def from_json(cls,json_str):
    data = json.loads(json_str)
    return cls(**data)

  def to_json(self, **options):
    return json.dumps(asdict(self),**options)  
  
  def reset(self):
    self.current_epoch = 0
    self.learning_loss = []
    self.validation_loss = []
    self.validation_accuracy = []

    

  def track_learning(self, epoch : int, batch : int,loss : float):
    self.learning_loss = grow(self.learning_loss, epoch, [])
    self.learning_loss[epoch] = grow(self.learning_loss[epoch], batch, None)
    self.learning_loss[epoch][batch] = loss

  def track_validation(self, epoch : int, loss : float, accuracy : float):
    self.validation_loss = grow(self.validation_loss, epoch, None)
    self.validation_accuracy = grow(self.validation_accuracy, epoch, None)
    self.validation_loss[epoch] = loss
    self.validation_accuracy[epoch] = accuracy

  def track_model_checkpoint(self, epoch : int, check_point_name : str):  
    pass


    