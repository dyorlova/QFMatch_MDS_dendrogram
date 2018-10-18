
import math
import point

import numpy as np
from scipy.stats import mstats


class _BaseBinner(object):
  """Abstract class - inheriting classess bin given list of points."""

  def __init__(self, points, min_points_per_bin=None):
    """Constructor.

    Args:
      points: list of point.Points objects.
      min_points_per_bin: desired min number of points in each bin.
    """
    self._points = list(points)
    self._min_points_per_bin = (
        min_points_per_bin or int(math.ceil(2 * math.log(len(points)))))
    print 'Min points per bin is %s' % self._min_points_per_bin
    self._bins = []

  def GetBins(self):
    raise NotImplementedError


class SplittingInHalfBinner(_BaseBinner):
  """Bins points in kd-tree building fashion.

  Algorithm:
    1) Among all the points find the dimension with max variance 
    2) Sort the points along the max var dimension
    3) Split points in half
    4) Recursively repeat 1,2,3 for left and right part
    5) Stop when division of number of points passed to the function divided
       by 2 is smaller than self._min_points_per_bin.
  """

  def __init__(self, *args, **kwargs):
    super(SplittingInHalfBinner, self).__init__(*args, **kwargs)
    self._splitting_min_max_coordinates = []

  def GetBins(self):
    if not self._bins:
      self._CalculateBins(self._points)
    total_size = sum([len(b.GetPoints()) for b in self._bins]) 
    assert total_size == len(self._points), '%s vs %s' % (
        total_size, len(self._points))
    return self._bins

  def GetSplittingCoordinates(self):
    """Returns list of coordinates which were used to bin data.
  
    Can be useful for binning visualization.
    """
    return self._splitting_min_max_coordinates

  def _GetVariances(self, points):
    return np.var(np.array([p.GetCoordinates() for p in points]), axis=0)

  def _GetCoordinateIndexWithMaxVariance(self, points):
    variances = self._GetVariances(points)
    max_variance = 0
    max_variance_coordinate_index = None
    for coordinate_index, variance in enumerate(variances):
      if variance >= max_variance:
        max_variance = variance
        max_variance_coordinate_index = coordinate_index
    return max_variance_coordinate_index

  def _CalculateBins(self, points, lower_borders=None, upper_borders=None):
    # lower_borders and upper_borders are useless for calculation but they
    # can be used to populate self._splitting_min_max_coordinates which
    # can be used to visualize binning process.

    if not lower_borders:
      lower_borders = []
      for i, _ in enumerate(points[0].GetCoordinates()):
        lower_borders.append(min(p.GetCoordinate(i) for p in points))

    if not upper_borders:
      upper_borders = []
      for i, _ in enumerate(points[0].GetCoordinates()):
        upper_borders.append(max(p.GetCoordinate(i) for p in points))

    median_index = len(points) / 2

    max_var_coordinate_index = self._GetCoordinateIndexWithMaxVariance(points)
    points.sort(key=lambda p: p.GetCoordinate(max_var_coordinate_index))
    
    if median_index <= self._min_points_per_bin:
      binn = Bin(points)
      binn.CalculateFixedMean()
      self._bins.append(binn)
    else:
      splitting_coordinates = []
      left_lower_borders = []
      right_lower_borders = []
      left_upper_borders = []
      right_upper_borders = []

      num_splitting_coordinates = int(
          math.pow(2, points[0].GetNumCoordinates() - 1))

      for i, _ in enumerate(points[0].GetCoordinates()):
        if i == max_var_coordinate_index:
          splitting_coordinates.append(
              tuple(points[median_index].GetCoordinate(i) 
                    for _ in xrange(num_splitting_coordinates)))
          left_lower_borders.append(lower_borders[i])
          left_upper_borders.append(points[median_index].GetCoordinate(i))
          right_lower_borders.append(points[median_index].GetCoordinate(i))
          right_upper_borders.append(upper_borders[i])
        else:
          coordinate = []
          # 0 0
          # 0 1
          # 1 0
          # 1 1
          # We need all combinations of lower and upper border across all 
          # dimensions to insert into splitting coordinates, e.g in 3D:
          # [0, 1, 0, 1]  [0, 0, 1, 1],  [median, median, median].
          # So looking at reversed binary form of each number from
          # 0 to 2^(n-1) we create all such combinations.
          for min_max_dec in xrange(num_splitting_coordinates):
            min_max_bin = list(reversed(bin(min_max_dec)[2:]))
            if i < max_var_coordinate_index:
              coordinate.append(
                  upper_borders[i] 
                  if len(min_max_bin) > i and min_max_bin[i] == '1'
                  else lower_borders[i])
            else:
              # If cur coordinate index is higher than max_coordinate_index, we 
              # need to check min_max_bin[i - 1].
              # For example in 3D we will have at most 4 numbers, thus the last
              # reversed binary repr we will look at is 11.
              # If max_var_coordinate is 1 then we need to compare coordinate
              # with index 0 against min_max_bin[0] and coordinate with index 2
              # against min_max_bin[2].
              coordinate.append(
                  upper_borders[i] 
                  if len(min_max_bin) > i - 1 and min_max_bin[i - 1] == '1' 
                  else lower_borders[i])

          splitting_coordinates.append(tuple(coordinate))

          left_lower_borders.append(lower_borders[i])
          left_upper_borders.append(upper_borders[i])
          right_lower_borders.append(lower_borders[i])
          right_upper_borders.append(upper_borders[i])

      self._splitting_min_max_coordinates.append(splitting_coordinates)

      left = points[:median_index]
      right = points[median_index:]
      self._CalculateBins(
          left, lower_borders=left_lower_borders,
          upper_borders=left_upper_borders)
      self._CalculateBins(
          right, lower_borders=right_lower_borders, 
          upper_borders=right_upper_borders)


class Bin(object):
  """Represents single bin."""

  def __init__(self, points=None):
    self._points = points or []
    self._gmean = None
    self._gmean_calculated_on_points_num = 0
    self._mean = None
    self._mean_calculated_on_points_num = 0
    # We call it fixed mean because IT DOES NOT ALWAYS
    # REPRESENT MEAN OF THE POINTS WHICH CURRENT Bin OBJECT CONTAINS.
    # Client can set it manually. It is used for example in situation
    # when one wants to keep mean value calculated on mixed data 
    # for bins after separation of the medley.
    self._fixed_mean = None
    self._CalculateGmean()
    
  def CalculateFixedMean(self):
    if self._fixed_mean is None:
      self._fixed_mean = np.mean(
          np.array([p.GetCoordinates() for p in self._points]), axis=0)  
    else:
      raise RuntimeError('Fixed mean is already calculated')

  def SetFixedMean(self, fixed_mean):
    self._fixed_mean = fixed_mean

  def GetMean(self):
    if len(self._points) == self._mean_calculated_on_points_num:
      return self._mean
    else:
      self._CalculateMean()
      return self._mean

  def GetGmean(self):
    if len(self._points) == self._gmean_calculated_on_points_num:
      return self._gmean
    else:
      self._CalculateGmean()
      return self._gmean

  def GetFixedMean(self):
    if self._fixed_mean is None:
      raise ValueError('Fixed mean is not calculated')
    else:
      return self._fixed_mean

  def GetPoints(self):
    return self._points

  def AddPoint(self, point):
    self._points.append(point)

  def _CalculateGmean(self):
    self._gmean = np.absolute(
        mstats.gmean(np.array([p.GetCoordinates() for p in self._points])))
    self._gmean_calculated_on_points_num = len(self._points)

  def _CalculateMean(self):
    self._mean = np.mean(
        np.array([p.GetCoordinates() for p in self._points]), axis=0)  
    self._mean_calculated_on_points_num = len(self._points)    
