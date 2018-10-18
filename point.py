class Point(object):
  """Represents single point in a dataset."""

  def __init__(self, coordinates, c_id):
    """Constructor.

    Args:
      coordinates: N-dimensional coordinates.
      c_id: cluster id corresponding to the point.
    """
    self._coordinates = coordinates
    self._cluster_id = c_id
    self._custom_attributes = {}

  def GetCoordinates(self):
    return self._coordinates

  def GetCoordinate(self, c_index):
    return self._coordinates[c_index]

  def GetClusterId(self):
    return self._cluster_id

  def GetNumCoordinates(self):
    return len(self._coordinates)

  def SetCustomAttribute(self, key, value):
    self._custom_attributes[key] = value

  def GetCustomAttribute(self, key):
    return self._custom_attributes.get(key)