from flexlearn import FileSystemTracker

def test_serialization():
  fs = FileSystemTracker("/tmp/foo_filesystem_tracker")
  fs.max_epochs = 27
  fs.save()

  fs2 = FileSystemTracker("/tmp/foo_filesystem_tracker")
  assert fs2.max_epochs == 27
  fs2.reset()

  fs3 = FileSystemTracker("/tmp/foo_filesystem_tracker")
  assert fs3.max_epochs == 25
  fs3.reset()
