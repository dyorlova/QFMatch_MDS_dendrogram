"""Contains cluster-related classes and utilities."""


class ClusterId(object):
  """Represent cluster id.

  Logically cluster ID can be a scalar or consist of multiple parts
  (e.g. when it represents id of two or more merged clusters).
  This class allows to encapsulate logic handling various types of
  cluster IDs.
  """

  def __init__(self, parts):
    if not parts:
      raise ValueError('No parts: %s' % parts)

    if isinstance(parts, (str, unicode, int, float)):
      self._parts = (str(parts),)
    else:
      self._parts = tuple(sorted(str(p) for p in parts))

    self._parts_as_set = set(self._parts)

  @classmethod
  def MergeFromTwo(cls, first, second):
    return cls.MergeFromMany([first, second])

  @classmethod
  def MergeFromMany(cls, iterable):
    parts = []
    for cluster_id in iterable:
      parts.extend(cluster_id.GetParts())
    return cls(parts)

  def SplitForEachPart(self):
    for part in self._parts:
      yield ClusterId([part])

  def IsNegative(self):
    return (len(self._parts) == 1
            and self._parts[0].isdigit() 
            and int(self._parts[0]) < 0)

  def IsZero(self):
    return self._parts == ('0',)

  def GetParts(self):
    return self._parts

  def HasPart(self, part):
    return part in self._parts_as_set

  def __eq__(self, other):
    if other is None:
      return False
    elif other is self:
      return True
    elif not isinstance(other, ClusterId):
      raise TypeError('other is of type %s' % type(other))
    else:
      return self._parts == other.GetParts()

  def __hash__(self):
    return hash(self._parts)

  def __str__(self):
    return '+'.join(self._parts)