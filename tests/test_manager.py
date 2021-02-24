from flexlearn import Manager

def test_learn():
  manager = Manager()
  manager.add_experiment("my_experiment", {})
  assert False == manager.learn("other_experiment")
  assert True == manager.learn("my_experiment")

  