import json
from .basic_tracker import BasicTracker
from typing import Dict, List,Tuple
from pathlib import Path
import shutil

class FileSystemTracker:
  basic_tracker = BasicTracker()
  
  def __init__(self, path):
    self.path = Path(path)
    self.path.mkdir(parents=True, exist_ok=True)
    if self.path.joinpath('index.json').exists():
      self.basic_tracker = BasicTracker.from_json(self.path.joinpath('index.json').read_text())

  @property
  def max_epochs(self) -> int:
    return self.basic_tracker.max_epochs

  @max_epochs.setter
  def max_epochs(self, value : int) -> None:
    self.basic_tracker.max_epochs = value  


  def reset(self):
    # TODO: erase files?
    self.basic_tracker.reset()
    shutil.rmtree(self.path)
    self.path.mkdir(parents=True, exist_ok=True)
    

  def track_learning(self, epoch : int, batch : int,loss : float):
    self.basic_tracker.track_learning(epoch, batch, loss)

  def track_validation(self, epoch : int, loss : float, accuracy : float):
    self.basic_tracker.track_validation(epoch, loss, accuracy)

  def track_model_checkpoint(self, epoch : int, check_point_name : str):  
    pass


  def save(self):
    self.path.joinpath('index.json').write_text(self.basic_tracker.to_json())
