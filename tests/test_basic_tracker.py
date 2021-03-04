from flexlearn import BasicTracker

def test_to_json():
  o = BasicTracker()
  o.track_learning(0,0, 0.1)
  json = """
{
    "current_epoch": 0,
    "learning_loss": [
        [
            0.1
        ]
    ],
    "max_epochs": 25,
    "validation_accuracy": [],
    "validation_loss": []
}

  """
  #print(o.to_json(sort_keys=True, indent=4))
  assert json.strip() == o.to_json(sort_keys=True, indent=4)

def test_from_json():
  json = """
{
    "current_epoch": 0,
    "learning_loss": [
        [
            0.1
        ]
    ],
    "max_epochs": 33,
    "validation_accuracy": [],
    "validation_loss": []
}

  """
  
  o = BasicTracker.from_json(json)
  assert 33 == o.max_epochs    