import collections
import re

import cluster
import point


class DataLoader(object):
  """Loads data from file to list of ND point.Point objects."""

  def __init__(self, filename, num_first_rows_to_skip=2, line_separator='\r', 
               x_columns=tuple(), cluster_id_column=2,
               cluster_ids_to_exclude=None,
               columns_separator_regex=r'\s'):

    assert cluster_id_column not in x_columns

    self._filename = filename
    self._num_first_rows_to_skip = num_first_rows_to_skip
    self._line_separator = line_separator
    self._columns_separator_regex = columns_separator_regex
    self._cluster_id_column = cluster_id_column
    self._cluster_ids_to_exclude = cluster_ids_to_exclude or set()
    self._x_columns = x_columns
    
    for column in self._x_columns:
      assert column >= 0
    assert self._cluster_id_column >= 0

  def LoadAndReturnPoints(self, point_custom_attributes=None):
    return list(self._OpenFileAndYieldPoints(
        point_custom_attributes=point_custom_attributes))

  def LoadAndReturnPointsDividedByClusterId(self, point_custom_attributes=None):
    points_by_cluster_id = collections.defaultdict(list)
    for point in self._OpenFileAndYieldPoints(
        point_custom_attributes=point_custom_attributes):      
      points_by_cluster_id[point.GetClusterId()].append(point)
    return points_by_cluster_id

  def _OpenFileAndYieldPoints(self, point_custom_attributes=None):
    with open(self._filename, 'r') as file_descr:
      for i, row in enumerate(file_descr.read().split(self._line_separator)): 
        row = row.strip()
        if i >= self._num_first_rows_to_skip and row: 
          try:
            data_list = [
                s for s in re.split(self._columns_separator_regex, row)]
            xs = [float(data_list[i]) for i in self._x_columns]
            if (data_list[self._cluster_id_column] 
                in self._cluster_ids_to_exclude):
              continue
            cluster_id = cluster.ClusterId(data_list[self._cluster_id_column])
          except (ValueError, TypeError):
            print 'Failed on processing row "%s"' % row
            raise
          else:
            cur_point = point.Point(tuple(xs), cluster_id)
            if point_custom_attributes:
              for key, value in point_custom_attributes.iteritems():
                cur_point.SetCustomAttribute(key, value)
            yield cur_point