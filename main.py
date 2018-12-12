# encoding=utf-8
"""Cluster matching.

Algorightm:
There are 2 data files with clustered 2d points: we call them "left" file
and "right" file in the scope of this project.
User can specify which cluster from the left file to find matches for.
Algorithm will take this cluster, mix it with all the clusters from the
right file. Bin the mix, then separate the mix into pairs
(left cluster, one right cluster) for each right cluster. Then apply the same
"bin grid" for each of the pairs and calculate dissimilarities using quadratic
form based comparison.
Unmatched clusters are attempted to be merged to the originally matched ones.
"""

import collections
import copy
import datetime
import math
import sys
from matplotlib import pyplot
import matplotlib.patches as mpatches
import numpy as np
from scipy.spatial import distance
from sklearn import manifold
import networkx
from networkx.drawing import nx_agraph

import binner
import cluster
import color_generator
import data_loader
import point

# SETTINGS.
# TODO: consider moving them to sys.args and let user provide them from CLI.
_LEFT_FILENAME = '/Users/dyorlova/Downloads/testQF/ManualLiveBALBc2.csv'
_RIGHT_FILENAME = '/Users/dyorlova/Downloads/testQF/ManualLiveBALBc2.csv'

# Minimal bin size for binning the mix.
_BIN_SIZE = 300

# How many first rows in data files contain bogus data (headers, description
# etc)
_NUM_FIRST_ROWS_TO_SKIP_IN_THE_DATA_FILES = 1
# Which character is used for next line in data files.
_DATA_FILES_LINE_SEPARATOR = '\r'
_COLUMNS_SEPARATOR_REGEX= r','
# In which columns we have features' values in data files.
_DATA_FILES_X_COLUMNS = (0, 1, 2, 3, 4, 5, 6, 7)
# In which column we have cluster id in data files.
_DATA_FILES_CLUSTER_ID_COLUMN = 8

# Whether we want not to show clusters with ids < 0 on plot.
_DO_NOT_SHOW_NEGATIVE_CLUSERS_ON_PLOT = False

# Coefficient which will be multiplied with left ad right cluster standart
# deviation to compare with median of right and left clusters respectively.
# If right cluster's median is laying within this variable multiplied
# with left cluster standart deviation OR vice versa (left cluster's median is
# within right cluster's
# std * _SIGMA_MULTIPLIER_TO_CONSIDER_CLUSTERS_WITH_MEDIAN_WITHIN)
# then right cluster and left cluster will be considered for matching.
_SIGMA_MULTIPLIER_TO_CONSIDER_CLUSTERS_WITH_MEDIAN_WITHIN = 3

# OTHER STRING LITERALS.
_DATASET_TYPE_CUSTOM_ATTRIBUTE_NAME = 'dataset_type'
_RIGHT_DATASET = 'right_dataset'
_LEFT_DATASET = 'left_dataset'

_CALCULATE_MDS_MODE_POINT = 'POINT'
_CALCULATE_MDS_MODE_CLUSTER_MEDIAN = 'CLUSTER_MEDIAN'
_CALCULATE_MDS_MODE_BIN_MEDIAN = 'BIN_MEDIAN'

_CALCULATE_MDS_MODE = _CALCULATE_MDS_MODE_CLUSTER_MEDIAN

assert _CALCULATE_MDS_MODE in {
    _CALCULATE_MDS_MODE_POINT, _CALCULATE_MDS_MODE_CLUSTER_MEDIAN,
    _CALCULATE_MDS_MODE_BIN_MEDIAN}

_MDS_PLOT_POINT_SIZE_MULTIPLIER = 1000


def _LoadPointsByClusterId(filename, cust_attrs_to_set=None):
  return data_loader.DataLoader(
      filename,
       num_first_rows_to_skip=
       _NUM_FIRST_ROWS_TO_SKIP_IN_THE_DATA_FILES,
       line_separator=_DATA_FILES_LINE_SEPARATOR,
       x_columns=_DATA_FILES_X_COLUMNS,
       cluster_id_column=_DATA_FILES_CLUSTER_ID_COLUMN,
       cluster_ids_to_exclude={'0', '-1000'},
       columns_separator_regex=_COLUMNS_SEPARATOR_REGEX
  ).LoadAndReturnPointsDividedByClusterId(
      point_custom_attributes=cust_attrs_to_set or {})


def _IsWithin(center_coordinates, point_coordinates, allowed_interval):
  assert len(center_coordinates) > 0
  assert len(center_coordinates) == len(point_coordinates) == len(
      allowed_interval)
  for i, center_coordinate in enumerate(center_coordinates):
    if (math.fabs(center_coordinate - point_coordinates[i])
        > allowed_interval[i]):
      return False
  return True


def _YieldAllSubsets(original_list):
  """Yields elements of power set of given set (technically it is given list).

  Logic is based on binary number representation of existence / absence of
  element in subset.
  E.g. for set [1, 2, 3] we have power set of power 8:
  {}, {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}.
  We can have 8 binary numbers [0-7] where each bit would represent existence /
  absence of value from original set in the subset.
  {} -> 000
  {1} -> 100
  {2} -> 010
  {3} -> 001
  {1, 2} -> 110
  {1, 3} -> 101
  {2, 3} -> 011
  {1, 2, 3} -> 111.
  For general case we need to go through pow(2, n) - 1 such numbers where n is
  power of original set.
  """
  for dec_num in xrange(int(math.pow(2, len(original_list)))):
    subset = set()
    for i, b in enumerate(reversed(bin(dec_num).replace('0b', '', 1))):
      if b == '1':
        subset.add(original_list[i])
    yield subset


class _BinsCollection(object):
  """Stores list of bins + metadata characterizing the whole collection.

  (I.e. total number of points in all bins).

  Public methods:
    def AddBin(self, bin_to_add): adds bin to collection.
    def GetBins(self): returns list of all bins.
    def GetBin(self, bin_index): returns bin at particular index
        (in order of bins adding to the collection - first added has index 0,
        second added - index 1 etc).
    def GetTotalNumPoints(self): returns sum of numbers of points across all
        bins in collection.
    def GetMedian(self): returns median across all points.
    def GetSigma(self): returns standard deviation across all points.
  """

  def __init__(self):
    # List of binner.Bin objects.
    self._bins = []
    # Total number of points in all bins.
    self._total_num_points = 0
    self._median = None
    self._sigma = None

  def AddBin(self, bin_to_add):
    """Adds bin to collection.

    Args:
      bin_to_add: binner.Bin object.
    """
    self._bins.append(bin_to_add)
    self._total_num_points += len(bin_to_add.GetPoints())
    self._median = None
    self._sigma = None

  def GetBins(self):
    return self._bins

  def GetBin(self, bin_index):
    return self._bins[bin_index]

  def GetTotalNumPoints(self):
    return self._total_num_points

  def _GetAllPointsCoordinates(self):
    all_coordinates = []
    for b in self._bins:
      for p in b.GetPoints():
        all_coordinates.append(p.GetCoordinates())
    return all_coordinates

  def GetMedian(self):
    if self._median is None:
      self._median = np.median(self._GetAllPointsCoordinates(), axis=0)
    return self._median

  def GetSigma(self):
    if self._sigma is None:
      self._sigma = np.std(self._GetAllPointsCoordinates(), axis=0)
    return self._sigma

  def _GenerateRelations(self, cluster_ids):
    if not cluster_ids:
      raise ValueError('No cluster ids')
    if len(cluster_ids) == 1:
      return
   
    # List of tuples: (distance between medians, dissimilarity).
    distances_and_dissimilarities = []
    # List of tuples: (closeness score, dissimilarity).
    closeness_scores_and_dissimilarities = []

    min_dissimilarity = None
    max_distance = None
    # Calculate diss between each pair of clusters.
    for i, first_cluster_id in enumerate(cluster_ids):
      for j in xrange(i + 1, len(cluster_ids)):
        second_cluster_id = cluster_ids[j]
        print 'Calculating dissimilarity between %s and %s' % (
            first_cluster_id, second_cluster_id)
        distance_between_medians = _Dist(
            self._bin_collection_by_cluster_id.get(
                first_cluster_id).GetMedian(),
            self._bin_collection_by_cluster_id.get(
                second_cluster_id).GetMedian())
        diss = _CalculateDissimilarityBetweenClusters(
            first_cluster_id, 
            self._bin_collection_by_cluster_id.get(first_cluster_id),
            second_cluster_id,
            self._bin_collection_by_cluster_id.get(second_cluster_id))
        if max_distance is None or distance_between_medians > max_distance:
          max_distance = distance_between_medians
        if (min_dissimilarity is None 
            or min_dissimilarity < diss.dissimilarity_score):
          min_dissimilarity = diss.dissimilarity_score
        distances_and_dissimilarities.append((distance_between_medians, diss))

    # Defines the metric to scale all distances between medians to the number
    # which order does not exceed the order of the min qf score to avoid the
    # correcting part in the formula below to outweight the QF part.
    closeness_scaling = (
        _DefineOrderOfTheNumber(max_distance) 
        / _DefineOrderOfTheNumber(min_dissimilarity))

    for distance_between_medians, diss in distances_and_dissimilarities:
      closeness_scores_and_dissimilarities.append((
          (diss.dissimilarity_score
           + (1.0 / closeness_scaling) * distance_between_medians), 
          diss))
      print '%s::%s' % (
          diss.dissimilarity_score,
          (diss.dissimilarity_score + 
           (1.0 / closeness_scaling) * distance_between_medians))

    paired_cluster_ids = set()
    next_level_cluster_ids = set()

    closeness_scores_and_dissimilarities = sorted(
        closeness_scores_and_dissimilarities, key=lambda (c, _): c)
    for _, diss in closeness_scores_and_dissimilarities:
      if (diss.left_cluster_id in paired_cluster_ids
          and diss.right_cluster_id in paired_cluster_ids):
        continue
      elif diss.left_cluster_id in paired_cluster_ids:
        # If this is the smallest dissimilarity between cluster A and B BUT
        # for cluster A there was smaller dissimilarity with cluster C earlier,
        # make sure that cluster B is not paired with any other cluster at this
        # tree level.
        paired_cluster_ids.add(diss.right_cluster_id)
        next_level_cluster_ids.add(diss.right_cluster_id)
      elif diss.right_cluster_id in paired_cluster_ids:
        paired_cluster_ids.add(diss.left_cluster_id)
        next_level_cluster_ids.add(diss.left_cluster_id)
      else:
        parent_cluster_id = cluster.ClusterId(
            (diss.left_cluster_id, diss.right_cluster_id))
        print 'Pairing clusters %s and %s' % (
            diss.left_cluster_id, diss.right_cluster_id)

        self._graph.add_node(str(parent_cluster_id))

        self._graph.add_edge(
            str(parent_cluster_id), str(diss.left_cluster_id))
        self._graph.add_edge(
            str(parent_cluster_id), str(diss.right_cluster_id))

        paired_cluster_ids.add(diss.right_cluster_id)
        paired_cluster_ids.add(diss.left_cluster_id)
        next_level_cluster_ids.add(parent_cluster_id)
        self._bin_collection_by_cluster_id[parent_cluster_id] = (
            _MixCollections(
                self._bin_collection_by_cluster_id.get(diss.left_cluster_id),
                self._bin_collection_by_cluster_id.get(diss.right_cluster_id)))
        self._node_sizes[str(parent_cluster_id)] = self._GetNodeSize(
            self._bin_collection_by_cluster_id.get(parent_cluster_id)
            .GetTotalNumPoints())

    # Recursion here can be avoided if needed 
    # (performance issues, max recursion depth etc).
    self._GenerateRelations(list(next_level_cluster_ids))


def _DefineOrderOfTheNumber(number):
  """Defines order of magnitude for a number.

  https://en.wikipedia.org/wiki/Order_of_magnitude.

  Args:
    number: number to define the order of magnitude for.

  Returns: 
    10 in the power of arg number's order of magnitude.
  """
  return 10 ** math.floor(math.log10(number))


class _Dissimilarity(object):
  """Structure containing information about dissimilarity between 2 clusters."""
  def __init__(self, left_cluster_id, right_cluster_id, dissimilarity_score):
    self.left_cluster_id = left_cluster_id
    self.right_cluster_id = right_cluster_id
    self.dissimilarity_score = dissimilarity_score

  def IsBetterThan(self, other):
    if not isinstance(other, _Dissimilarity):
      raise TypeError('other is of type %s' % type(other))

    return self.dissimilarity_score < other.dissimilarity_score

  def IsBetterThanOrSame(self, other):
    if not isinstance(other, _Dissimilarity):
      raise TypeError('other is of type %s' % type(other))

    return self.dissimilarity_score <= other.dissimilarity_score

  def __eq__(self, other):
    if other is None:
      return False
    elif other is self:
      return True
    else:
      return (
          other.left_cluster_id == self.left_cluster_id
          and other.right_cluster_id == self.right_cluster_id
          and other.dissimilarity_score == self.dissimilarity_score)

  def __hash__(self):
    return hash((self.left_cluster_id, self.right_cluster_id,
                 self.dissimilarity_score))


class _Matcher(object):
  """Matches clusters.

  Concepts of "left" and "right" entity correspond to first (left) file and
  second (right) file which we match clusters for.
  Concept of "mix" / "mixed" entity means that it somehow includes information
  from left and right files (i.e. "Bin" contains points from both files).
  """

  def __init__(self):
    # Most variables below are actually collections. They are initialized with
    # Nones to guarantee TypeError in methods (private and public) which will
    # attempt to utilize them before actually setting / filling them. The
    # intention is to minimize chances of coding errors leading to false
    # positive "success" runs.
    # The other (better) option would be to write automatic (e.g. unit) tests
    # but since this code is not going to be productionized, there is laziness
    # preventing us from doing it.

    # Dict with cluster id as key and all points (points.Point objects)
    # related to this cluster as value in LEFT (base) file.
    self._all_left_points_by_cluster_id = None

    # Dict with cluster id as key and all points (points.Point objects)
    # related to this cluster as value in RIGHT (matching) file.
    self._all_right_points_by_cluster_id = None

    # List of bins after binning on mixed dataset.
    self._mix_bins = None

    # Dict with cluster id as key and _BinsCollection object with bins from
    # this right cluster as value.
    self._right_bin_collection_by_cluster_id = None

    # Dict with cluster id as key and _BinsCollection object with bins from
    # this left cluster as value.
    self._left_bin_collection_by_cluster_id = None

    # Actually a dict where key is tuple (left_cluster_id, right_cluster_id) and
    # the value is the actual _Dissimilarity object.
    self._dissimilarities = None

    # Actually a list of tuples (left_cluster_id, right_cluster_id).
    self._matched_pairs = None

    # Actually a dict where key is originally matched left cluster id and value
    # is a list of right cluster ids which were not matched with any left
    # cluster originally but which closest left cluster id was this dict's key.
    self._unmatched_right_by_closest_left_cluster_id = None

    # Actually a dict where key is originally matched right cluster id and value
    # is a list of left cluster ids which were not matched with any right
    # cluster originally but which closest right cluster id was this dict's key.
    self._unmatched_left_by_closest_right_cluster_id = None

    # Max distance between means of bins on mixed set of points.
    self._max_distance_between_bins = None

  def MatchAndMds(self):
    """Matches clusters, runs multi-dimensional scaling, draws it's results."""
    self._RunMatchingProcess()
    if _CALCULATE_MDS_MODE == _CALCULATE_MDS_MODE_CLUSTER_MEDIAN:
      self._MdsOnClusterMedian()
    elif _CALCULATE_MDS_MODE == _CALCULATE_MDS_MODE_BIN_MEDIAN:
      self._MdsOnBinMedian()
    elif _CALCULATE_MDS_MODE == _CALCULATE_MDS_MODE_POINT:
      self._Mds()
    else:
      raise ValueError('Unknown mds calculation mode %s' % _CALCULATE_MDS_MODE)

  def MatchAndDrawMatchedPoints(self):
    """Matches clusters, draws both samples with matched clusters having same
    color."""
    self._RunMatchingProcess()
    self._Draw2DGraphs()

  def MatchAndReturnDissimilarities(self):
    """Matches clusters and returns all dissimilarities encountered during
    matching."""
    self._RunMatchingProcess()
    return copy.deepcopy(self._dissimilarities)

  def _RunMatchingProcess(self):
    dt = datetime.datetime.now()
    self._LoadLeft()
    self._LoadRight()
    self._MixAndBin()
    self._SeparateMixedBins()
    self._CalculateMaxDistanceBetweenBins()
    self._CalculateDissimilarities()
    self._Match()
    self._ExhaustiveMerge()
    print 'Took %s' % (datetime.datetime.now() - dt).total_seconds()

  def _LoadLeft(self):
    # Load left points and mark each of it as 'left point' to be able
    # later to separate them after binning.
    cust_attrs_to_set = {_DATASET_TYPE_CUSTOM_ATTRIBUTE_NAME: _LEFT_DATASET}
    self._all_left_points_by_cluster_id = _LoadPointsByClusterId(
        _LEFT_FILENAME,
        cust_attrs_to_set=cust_attrs_to_set)
    print 'Left points are loaded. Clusters are %s' % (', ').join(
        [str(s) for s in self._all_left_points_by_cluster_id.iterkeys()])

  def _LoadRight(self):
    # Load right points and mark each of it as 'right point' to be able
    # later to separate them after binning.
    cust_attrs_to_set = {_DATASET_TYPE_CUSTOM_ATTRIBUTE_NAME: _RIGHT_DATASET}
    self._all_right_points_by_cluster_id = _LoadPointsByClusterId(
        _RIGHT_FILENAME, cust_attrs_to_set=cust_attrs_to_set)
    print 'Right points are loaded. Clusters are %s' % (', ').join(
        [str(s) for s in self._all_right_points_by_cluster_id.iterkeys()])

  def _MixAndBin(self):
    # Mix left and right files.
    the_mix = []
    for dict_of_points_divided_by_cluster_id in [
        self._all_right_points_by_cluster_id,
        self._all_left_points_by_cluster_id]:
      for points in dict_of_points_divided_by_cluster_id.itervalues():
        for p in points:
          the_mix.append(p)

    # And bin the medley.
    good_binner = binner.SplittingInHalfBinner(
        the_mix, min_points_per_bin=_BIN_SIZE)
    self._mix_bins = good_binner.GetBins()

  def _SeparateMixedBins(self):
    self._right_bin_collection_by_cluster_id = collections.defaultdict(
        _BinsCollection)
    self._left_bin_collection_by_cluster_id = collections.defaultdict(
        _BinsCollection)
    # Separate the medley keeping the same bin borders as were calculated on the
    # medley.
    for mix_bin in self._mix_bins:
      self._SeparateMixedBin(mix_bin)

  def _SeparateMixedBin(self, mix_bin):
    left_bin_by_cluster_id = {}
    right_bin_by_cluster_id = {}

    # Each bin from mixed set can potentially have points related
    # to each cluster from left and right dataset.
    # Here we will create Bin object corresponding to the bin from mixed set
    # for each left cluster and each right cluster. If there are no points
    # related to particular left or right cluster in this mix bin, then this new
    # Bin object will have no points in it.
    for c_id in self._all_left_points_by_cluster_id.iterkeys():
      left_bin = binner.Bin()
      # In the bin containing only left points keep the mean which was
      # calculated on the mixed bin.
      left_bin.SetFixedMean(mix_bin.GetFixedMean())
      left_bin_by_cluster_id[c_id] = left_bin

    for c_id in self._all_right_points_by_cluster_id.iterkeys():
      right_bin = binner.Bin()
      # In the bin containing only right points keep the mean which was
      # calculated on the mixed bin.
      right_bin.SetFixedMean(mix_bin.GetFixedMean())
      right_bin_by_cluster_id[c_id] = right_bin

    # Do the actual separation of bin with mixed points.
    for cur_point in mix_bin.GetPoints():
      if cur_point.GetCustomAttribute(
          _DATASET_TYPE_CUSTOM_ATTRIBUTE_NAME) == _LEFT_DATASET:
        left_bin_by_cluster_id[cur_point.GetClusterId()].AddPoint(cur_point)
      elif cur_point.GetCustomAttribute(
          _DATASET_TYPE_CUSTOM_ATTRIBUTE_NAME) == _RIGHT_DATASET:
        right_bin_by_cluster_id[cur_point.GetClusterId()].AddPoint(cur_point)
      else:
        raise ValueError(
            'Can not define which dataset point %s belongs to' % cur_point)

    for right_cluster_id, right_bin in right_bin_by_cluster_id.iteritems():
      self._right_bin_collection_by_cluster_id[right_cluster_id].AddBin(
          right_bin)

    for left_cluster_id, left_bin in left_bin_by_cluster_id.iteritems():
      self._left_bin_collection_by_cluster_id[left_cluster_id].AddBin(
          left_bin)

  def _CalculateMaxDistanceBetweenBins(self):
    """Calculate max distance between two farthest-apart mixed bins."""
    total_ops = len(self._mix_bins) * len(self._mix_bins)
    print 'Calculating max distance between bins. Total calculations: %s' % (
        total_ops)
    self._max_distance_between_bins = 0

    remove_prev_line_from_stdout = False

    for i, bin_i in enumerate(self._mix_bins):
      for j, bin_j in enumerate(self._mix_bins):

        cur_iter = i * len(self._mix_bins) + j
        if not cur_iter % 10000:
          if remove_prev_line_from_stdout:
            sys.stdout.write("\033[F")
          print 'Current iteration is %s out of %s' % (cur_iter, total_ops)
          remove_prev_line_from_stdout = True

        d = _Dist(bin_i.GetFixedMean(), bin_j.GetFixedMean())
        if self._max_distance_between_bins < d:
          self._max_distance_between_bins = d

    if remove_prev_line_from_stdout:
      sys.stdout.write("\033[F")
    print 'Max distance is calculated'

  def _CalculateDissimilarities(self):
    """Calculates dissimilarities between each left and right clusters."""
    self._dissimilarities = {}
    print 'Calculating dissimilarities'
    num_bins = len(self._mix_bins)

    for left_cluster_id, left_bin_collection in (self
        ._left_bin_collection_by_cluster_id.iteritems()):
      for right_cluster_id, right_bin_collection in (self
          ._right_bin_collection_by_cluster_id.iteritems()):
        if (
          _IsWithin(
              left_bin_collection.GetMedian(),
              right_bin_collection.GetMedian(),
              left_bin_collection.GetSigma()
              * _SIGMA_MULTIPLIER_TO_CONSIDER_CLUSTERS_WITH_MEDIAN_WITHIN)
          or
          _IsWithin(
              right_bin_collection.GetMedian(),
              left_bin_collection.GetMedian(),
              right_bin_collection.GetSigma()
              * _SIGMA_MULTIPLIER_TO_CONSIDER_CLUSTERS_WITH_MEDIAN_WITHIN)):
          # This operation can be easily parallelized via multiprocessing.
          d = _CalculateDissimilarityBetweenClusters(
              left_cluster_id, left_bin_collection, right_cluster_id,
              right_bin_collection)
          print 'Left cluster: %s, Right cluster: %s, dissimilarity: %s' % (
              d.left_cluster_id, d.right_cluster_id, d.dissimilarity_score)
          self._CaptureDissimilarity(d)
        else:
          print ('Left cluster %s median is not within %s sigma from right '
                 'cluster %s median and vice versa. Dissimilarity won\'t be '
                 'calculated') % (
                     left_cluster_id,
                     _SIGMA_MULTIPLIER_TO_CONSIDER_CLUSTERS_WITH_MEDIAN_WITHIN,
                     right_cluster_id)

    print 'Dissimilarities are calculated'


  def _Match(self):
    """Match clusters."""
    self._matched_pairs = []
    # Key is left cluster id, value is _Dissimilarity object containing
    # information about dissimilarity between given left cluster
    # and closest right cluster.
    closest_for_left = {}
    # Key is right cluster id, value is _Dissimilarity object containing
    # information about dissimilarity between given right cluster
    # and closest left cluster.
    closest_for_right = {}

    for diss in self._IterateThroughDissmilarities():
      cur_diss = closest_for_left.get(diss.left_cluster_id)
      if not cur_diss or diss.IsBetterThan(cur_diss):
        closest_for_left[diss.left_cluster_id] = diss

      cur_diss = closest_for_right.get(diss.right_cluster_id)
      if not cur_diss or diss.IsBetterThan(cur_diss):
        closest_for_right[diss.right_cluster_id] = diss

    # Now - find trivial matches for left and right clusters. Leave (in
    # "closest" dicts) only clusters which we were not able to find matches for.
    for left_cluster_id, diss in closest_for_left.items():
      # Make sure that right cluster closest to left cluster X and left
      # cluster closest to right cluster X match. left.closest = right and
      # right.closest = left.
      if diss.right_cluster_id in closest_for_right:
        if (closest_for_right[diss.right_cluster_id].left_cluster_id
            == left_cluster_id):
          print ('Left cluster: %s. Closest right cluster: %s. '
                 'Closest left cluster for the right cluster: %s. Matches.') % (
                     left_cluster_id, diss.right_cluster_id,
                     closest_for_right[diss.right_cluster_id].left_cluster_id)
          self._matched_pairs.append((left_cluster_id, diss.right_cluster_id))
          # We found the pairs for these clusters, delete them from closests
          # dicts.
          del closest_for_right[diss.right_cluster_id]
          del closest_for_left[left_cluster_id]
        else:
          print ('Left cluster: %s. Closest right cluster: %s. '
                 'Closest left cluster for the right cluster: %s. '
                 'Does not match.') % (
                     left_cluster_id, diss.right_cluster_id,
                     closest_for_right[diss.right_cluster_id].left_cluster_id)
      else:
        # Right cluster was already matched to another cluster before.
        # It likely means that there were 2 left clusters which dissimilarity
        # was smallest with the same right cluster.
        print ('Left cluster: %s. Closest right cluster: %s. '
               'Already found the match for right cluster.') % (
                   left_cluster_id, diss.right_cluster_id)

    # These ones are used in exhaustive merging.
    self._unmatched_right_by_closest_left_cluster_id = collections.defaultdict(
        list)
    self._unmatched_left_by_closest_right_cluster_id = collections.defaultdict(
        list)

    for right_cluster_id, diss in closest_for_right.iteritems():
      self._unmatched_right_by_closest_left_cluster_id[
          diss.left_cluster_id].append(right_cluster_id)
    for left_cluster_id, diss in closest_for_left.iteritems():
      self._unmatched_left_by_closest_right_cluster_id[
          diss.right_cluster_id].append(left_cluster_id)

    print 'Non-matched left clusters: %s' % ', '.join(
        [str(c) for c in closest_for_left.iterkeys()])
    print 'Non-matched right clusters: %s' % ', '.join(
        [str(c) for c in closest_for_right.iterkeys()])
    print 'Initially matched: %s' % [
        (str(first), str(second)) for first, second in self._matched_pairs]

  def _ExhaustiveMerge(self):
    self._ExhaustiveMergeProcessLeftUnmatchedClusters()
    self._ExhaustiveMergeProcessRightUnmatchedClusters()

  def _ExhaustiveMergeProcessRightUnmatchedClusters(self):
    iterator = self._unmatched_right_by_closest_left_cluster_id.iteritems()
    for maybe_matched_left_cluster_id, right_unmatched_cluster_ids in iterator:
      matched_right_cluster_id = None
      matched_pair_index = None
      matched_left_cluster_id = None

      for i, (l_id, r_id) in enumerate(self._matched_pairs):
        if l_id == maybe_matched_left_cluster_id:
          # This is right cluster initially matched to matched_left_cluster_id.
          matched_right_cluster_id = r_id
          # Find the index of original matched pair to be able to remove it from
          # the list of final matched pairs if we find the merged clusters with
          # better qf score.
          matched_pair_index = i
          matched_left_cluster_id = maybe_matched_left_cluster_id
          break

      if not matched_left_cluster_id:
        continue

      left_unmatched_cluster_ids = (
          self._unmatched_left_by_closest_right_cluster_id[
              matched_right_cluster_id])

      best_diss = self._ExhaustiveMergeOnSinglePair(
          matched_left_cluster_id, matched_right_cluster_id,
          left_unmatched_cluster_ids, right_unmatched_cluster_ids)

      if (best_diss.left_cluster_id != matched_left_cluster_id
          or best_diss.right_cluster_id != matched_right_cluster_id):
        del self._matched_pairs[matched_pair_index]
        self._matched_pairs.append(
            (best_diss.left_cluster_id, best_diss.right_cluster_id))

  def _ExhaustiveMergeProcessLeftUnmatchedClusters(self):
    iterator = self._unmatched_left_by_closest_right_cluster_id.iteritems()
    for maybe_matched_right_cluster_id, left_unmatched_cluster_ids in iterator:
      matched_right_cluster_id = None
      matched_pair_index = None
      matched_left_cluster_id = None

      for i, (l_id, r_id) in enumerate(self._matched_pairs):
        if r_id == maybe_matched_right_cluster_id:
          # This is left cluster initially matched to matched_right_cluster_id.
          matched_left_cluster_id = l_id
          # Find the index of origial matched pair to be able to remove it from
          # the list of final matched pairs if we find the merged clusters with
          # better qf score.
          matched_pair_index = i
          matched_right_cluster_id = maybe_matched_right_cluster_id
          break

      if not matched_right_cluster_id:
        continue

      right_unmatched_cluster_ids = (
          self._unmatched_right_by_closest_left_cluster_id[
              matched_left_cluster_id])

      best_diss = self._ExhaustiveMergeOnSinglePair(
          matched_left_cluster_id, matched_right_cluster_id,
          left_unmatched_cluster_ids, right_unmatched_cluster_ids)

      if (best_diss.left_cluster_id != matched_left_cluster_id
          or best_diss.right_cluster_id != matched_right_cluster_id):
        del self._matched_pairs[matched_pair_index]
        self._matched_pairs.append(
            (best_diss.left_cluster_id, best_diss.right_cluster_id))

  def _ExhaustiveMergeOnSinglePair(
      self, matched_left_cluster_id, matched_right_cluster_id,
      left_unmatched_cluster_ids, right_unmatched_cluster_ids):
    # List of left cluster ids which were not matched to any right cluster
    # initially and which closest cluster on the right is the
    # matched_right_cluster_id.
    original_best_diss = self._GetDissimilarity(matched_left_cluster_id,
                                                matched_right_cluster_id)

    left_merging_candidates = list(left_unmatched_cluster_ids)
    right_merging_candidates = list(right_unmatched_cluster_ids)

    print ('Starting exhaustive merging procedure. Original match: left %s, '
           'right %s. Left candidates are %s. Right candidates are %s') % (
               original_best_diss.left_cluster_id,
               original_best_diss.right_cluster_id,
               [str(c) for c in left_merging_candidates],
               [str(c) for c in right_merging_candidates])

    cur_best_diss = original_best_diss
    iteration = 0
    left_visited = set()
    right_visited = set()

    while left_merging_candidates or right_merging_candidates:
      if iteration > 0:
        print ('Continue exhaustive merging procedure. Previous best match: '
               'left %s, right %s. Left candidates are %s. Right candidates '
               'are %s') % (
                   cur_best_diss.left_cluster_id,
                   cur_best_diss.right_cluster_id,
                   [str(c) for c in left_merging_candidates],
                   [str(c) for c in right_merging_candidates])
      iteration += 1

      for cur_left_merging_candidates in _YieldAllSubsets(
          left_merging_candidates):
        for cur_right_merging_candidates in _YieldAllSubsets(
            right_merging_candidates):
          merged_left_cluster_id = cluster.ClusterId.MergeFromMany(
              list(cur_left_merging_candidates)
              + [original_best_diss.left_cluster_id])
          merged_right_cluster_id = cluster.ClusterId.MergeFromMany(
              list(cur_right_merging_candidates)
              + [original_best_diss.right_cluster_id])

          if ((merged_left_cluster_id, merged_right_cluster_id)
              == (original_best_diss.left_cluster_id,
                  original_best_diss.right_cluster_id)):
            continue

          left_bin_collection = self._MixLeftBinCollections(
              list(cur_left_merging_candidates)
              + [original_best_diss.left_cluster_id])
          right_bin_collection = self._MixRightBinCollections(
              list(cur_right_merging_candidates)
              + [original_best_diss.right_cluster_id])

          print 'Calculating dissimilarity for left %s and right %s' % (
              merged_left_cluster_id, merged_right_cluster_id)

          new_diss = _CalculateDissimilarityBetweenClusters(
              merged_left_cluster_id, left_bin_collection,
              merged_right_cluster_id, right_bin_collection)
          self._CaptureDissimilarity(new_diss)

          left_visited.update(cur_left_merging_candidates)
          right_visited.update(cur_right_merging_candidates)

          if new_diss.IsBetterThanOrSame(cur_best_diss):
            print ('Dissimilarity score %s for left %s and right %s is better '
                   'than current best score %s for left %s and right %s') % (
                       new_diss.dissimilarity_score,
                       merged_left_cluster_id,
                       merged_right_cluster_id,
                       cur_best_diss.dissimilarity_score,
                       cur_best_diss.left_cluster_id,
                       cur_best_diss.right_cluster_id)
            cur_best_diss = new_diss

      if cur_best_diss == original_best_diss:
        break
      elif cur_best_diss.IsBetterThanOrSame(original_best_diss):
        left_merging_candidates = []
        right_merging_candidates = []

        for left_part_cluster_id in (cur_best_diss.left_cluster_id
                                     .SplitForEachPart()):
          if (left_part_cluster_id
              in self._unmatched_right_by_closest_left_cluster_id):
            right_merging_candidates.extend(
                c for c in self._unmatched_right_by_closest_left_cluster_id[
                    left_part_cluster_id]
                if c not in right_visited)

        for right_part_cluster_id in (cur_best_diss.right_cluster_id
                                      .SplitForEachPart()):
          if (right_part_cluster_id
              in self._unmatched_left_by_closest_right_cluster_id):
             left_merging_candidates.extend(
                c for c in self._unmatched_left_by_closest_right_cluster_id[
                    right_part_cluster_id]
                if c not in left_visited)

        original_best_diss = cur_best_diss

    return cur_best_diss

  @staticmethod
  def _MixSomeSideBinCollections(bin_collections_by_cluster_id, cluster_ids):
    if not cluster_ids:
      raise ValueError('Nothing to mix. cluster_ids is %s' % cluster_ids)
    if not bin_collections_by_cluster_id:
      raise ValueError('BIN collections were not defined yet.')

    mixed = None
    for cluster_id in cluster_ids:
      if cluster_id in bin_collections_by_cluster_id:
        mixed = (
            bin_collections_by_cluster_id[cluster_id]
            if mixed is None
            else _MixCollections(mixed,
                                 bin_collections_by_cluster_id[cluster_id]))
      else:
        for cluster_id_part in cluster_id.SplitForEachPart():
          mixed = (
              bin_collections_by_cluster_id[cluster_id_part]
              if mixed is None
              else _MixCollections(
                  mixed, bin_collections_by_cluster_id[cluster_id_part]))
    return mixed

  def _MixLeftBinCollections(self, cluster_ids):
    return self._MixSomeSideBinCollections(
        self._left_bin_collection_by_cluster_id, cluster_ids)

  def _MixRightBinCollections(self, cluster_ids):
    return self._MixSomeSideBinCollections(
        self._right_bin_collection_by_cluster_id, cluster_ids)

  def _IterateThroughDissmilarities(self):
    for d in self._dissimilarities.itervalues():
      yield d

  def _CaptureDissimilarity(self, diss):
    self._dissimilarities[(diss.left_cluster_id, diss.right_cluster_id)] = diss

  def _GetDissimilarity(self, left_cluster_id, right_cluster_id):
    """Returns existing dissmilarity."""
    return self._dissimilarities[(left_cluster_id, right_cluster_id)]

  def _GetColorsByClusterId(self):
    chunk_ids_for_color_generation = [
        i for i in xrange(len(self._matched_pairs))]
    color_gen = color_generator.ColorGenerator(
        chunk_ids_for_color_generation,
        exclude_colors=[
            color_generator.KELLY_COLORS_BY_COLOR_NAME[
                color_generator.STRONG_BLUE]])
    colors_by_left_cluster_id = {}
    colors_by_right_cluster_id = {}
    for i, match in enumerate(self._matched_pairs):
      left_color = color_gen.GetColor(i)
      for cluster_id in match[0].SplitForEachPart():
        colors_by_left_cluster_id[cluster_id] = left_color

      right_color = color_gen.GetColor(i)
      for cluster_id in match[1].SplitForEachPart():
        colors_by_right_cluster_id[cluster_id] = right_color
    return colors_by_left_cluster_id, colors_by_right_cluster_id

  def _Draw2DGraphs(self):

    class _2DPlotData(object):
      """Struct to store data required to show 2d plots for one 2D projection.
      """

      def __init__(self, x_column, y_column):
        self.x_column = x_column
        self.y_column = y_column
        self.left_xs = []
        self.left_ys = []
        self.right_xs = []
        self.right_ys = []

    left_colors = []
    left_sizes = []

    right_colors = []
    right_sizes = []

    two_d_plots = collections.OrderedDict()
    for cluster_id, points in self._all_left_points_by_cluster_id.iteritems():
      for cur_point in points:
        for i_coordinate in xrange(0, cur_point.GetNumCoordinates()):
          for j_coordinate in xrange(i_coordinate + 1, cur_point.GetNumCoordinates()):
            two_d_plots[(i_coordinate, j_coordinate)] = _2DPlotData(
                  i_coordinate, j_coordinate)
        # The assumption is that all points have same number of coordinates,
        # so we know how many plots we will show from looking at any single
        # point and number of it's coordinates.
        break

    colors_by_left_cluster_id, colors_by_right_cluster_id = (
        self._GetColorsByClusterId())

    for cluster_id, points in self._all_left_points_by_cluster_id.iteritems():
      for cur_point in points:
        if (_DO_NOT_SHOW_NEGATIVE_CLUSERS_ON_PLOT
            and cur_point.GetClusterId().IsNegative()):
          continue
        elif cur_point.GetClusterId().IsZero():
          continue
        else:
          for i, x_value in enumerate(cur_point.GetCoordinates()):
            for j in xrange(i + 1, cur_point.GetNumCoordinates()):
              two_d_plots[(i, j)].left_xs.append(x_value)
              two_d_plots[(i, j)].left_ys.append(cur_point.GetCoordinate(j))
          if cur_point.GetClusterId() in colors_by_left_cluster_id:
            left_colors.append(colors_by_left_cluster_id[
                cur_point.GetClusterId()])
            left_sizes.append(20)
          else:
            left_colors.append(
                color_generator.GetKellyColor(color_generator.STRONG_BLUE))
            left_sizes.append(7)

    for cluster_id, points in self._all_right_points_by_cluster_id.iteritems():
      for cur_point in points:
        if (_DO_NOT_SHOW_NEGATIVE_CLUSERS_ON_PLOT
            and cur_point.GetClusterId().IsNegative()):
          continue
        elif cur_point.GetClusterId().IsZero():
          continue
        else:
          for i, x_value in enumerate(cur_point.GetCoordinates()):
            for j in xrange(i + 1, cur_point.GetNumCoordinates()):
              two_d_plots[(i, j)].right_xs.append(x_value)
              two_d_plots[(i, j)].right_ys.append(cur_point.GetCoordinate(j))
          if cur_point.GetClusterId() in colors_by_right_cluster_id:
            right_colors.append(
                colors_by_right_cluster_id[cur_point.GetClusterId()])
            right_sizes.append(20)
          else:
            right_colors.append(
                color_generator.GetKellyColor(color_generator.STRONG_BLUE))
            right_sizes.append(7)

    fig = pyplot.figure()

    left_patches = []
    for c_id, color in colors_by_left_cluster_id.iteritems():
      if c_id.IsZero():
        continue
      elif c_id.IsNegative() and _DO_NOT_SHOW_NEGATIVE_CLUSERS_ON_PLOT:
        continue
      else:
        patch = mpatches.Patch(color=color, label=str(c_id))
        left_patches.append(patch)

    right_patches = []
    for c_id, color in colors_by_right_cluster_id.iteritems():
      if c_id.IsZero():
        continue
      elif c_id.IsNegative() and _DO_NOT_SHOW_NEGATIVE_CLUSERS_ON_PLOT:
        continue
      else:
        patch = mpatches.Patch(color=color, label=str(c_id))
        right_patches.append(patch)

    for plot_row, two_d_plot_data in enumerate(two_d_plots.itervalues()):
      left_ax = fig.add_subplot(len(two_d_plots), 2, plot_row * 2 + 1)

      left_ax.scatter(
          two_d_plot_data.left_xs, two_d_plot_data.left_ys, c=left_colors,
          s=left_sizes)
      left_ax.legend(handles=left_patches, loc=4)

      right_ax = fig.add_subplot(len(two_d_plots), 2, plot_row * 2 + 2)

      right_ax.scatter(
          two_d_plot_data.right_xs, two_d_plot_data.right_ys, c=right_colors,
          s=right_sizes)
      right_ax.legend(handles=right_patches, loc=4)

    pyplot.show()

  def _MdsOnBinMedian(self):
    num_left_points = sum(
        len(v) for v in self._all_left_points_by_cluster_id.itervalues())
    num_right_points = sum(
        len(v) for v in self._all_right_points_by_cluster_id.itervalues())
    colors_by_left_cluster_id, colors_by_right_cluster_id = (
        self._GetColorsByClusterId())

    # Parallel arrays.
    coordinates = []
    cluster_ids = []

    # Number of bin medians added to coordinates for 'Left' side.
    num_left_coordinates = 0

    # Mix both sides and run mds.
    for bin_collection_by_cluster_id, num_total_points, side in (
        (self._left_bin_collection_by_cluster_id, num_left_points, 'Left'),
        (self._right_bin_collection_by_cluster_id, num_right_points, 'Right')):
      for cluster_id, bin_collection in (bin_collection_by_cluster_id
                                        .iteritems()):
        for cur_bin in bin_collection.GetBins():
          if cur_bin.GetPoints():
            coordinates.append(
                np.median(
                    np.array([p.GetCoordinates() for p in cur_bin.GetPoints()]),
                    axis=0))
            cluster_ids.append(cluster_id)
            if side == 'Left':
              num_left_coordinates += 1

    mds = manifold.MDS(n_components=2)
    print 'Running mds'
    result = mds.fit_transform(coordinates).tolist()
    print 'Done with mds'

     # Split mds coordinates back to left and right.
    left_mds_coordinates_per_cluster_id = collections.defaultdict(list)
    right_mds_coordinates_per_cluster_id = collections.defaultdict(list)
    for i, (x, y) in enumerate(result):
      if i < num_left_coordinates:
        left_mds_coordinates_per_cluster_id[cluster_ids[i]].append((x, y))
      else:
        right_mds_coordinates_per_cluster_id[cluster_ids[i]].append((x, y))

    # Define data to output at the plot.
    print 'Building subplots for left side'
    left_xs, left_ys, left_colors, left_sizes, left_patches = (
        self._GetMdsSubplotDataForOneSide(
            num_left_coordinates,
            left_mds_coordinates_per_cluster_id,
            colors_by_left_cluster_id))
    print 'Building subplots for right side'
    right_xs, right_ys, right_colors, right_sizes, right_patches = (
        self._GetMdsSubplotDataForOneSide(
            len(coordinates) - num_left_coordinates,
            right_mds_coordinates_per_cluster_id,
            colors_by_right_cluster_id))

    # Calculate axis limits for both subplots.
    x_lim, y_lim = self._DefinePlotLimits(
        left_xs + right_xs, left_ys + right_ys)

    # Draw the plot.
    fig = pyplot.figure()
    left_ax = fig.add_subplot(121)

    left_ax.scatter(left_xs, left_ys, c=left_colors, s=left_sizes)
    left_ax.set_xlim(x_lim)
    left_ax.set_ylim(y_lim)
    left_ax.legend(handles=left_patches)

    right_ax = fig.add_subplot(122)

    right_ax.scatter(right_xs, right_ys, c=right_colors, s=right_sizes)
    right_ax.set_xlim(x_lim)
    right_ax.set_ylim(y_lim)
    right_ax.legend(handles=right_patches)

    pyplot.suptitle('MDS on bin medians')

    pyplot.show()

  def _MdsOnClusterMedian(self):
    num_left_points = sum(
        len(v) for v in self._all_left_points_by_cluster_id.itervalues())
    num_right_points = sum(
        len(v) for v in self._all_right_points_by_cluster_id.itervalues())
    colors_by_left_cluster_id, colors_by_right_cluster_id = (
        self._GetColorsByClusterId())

    # Parallel arrays.
    coordinates = []
    cluster_ids = []
    ratio_in_total = []

    # Mix both sides and run mds.
    for points_by_cluster_id, num_total_points, side in (
        (self._all_left_points_by_cluster_id, num_left_points, 'Left'),
        (self._all_right_points_by_cluster_id, num_right_points, 'Right')):
      for cluster_id, cur_points in points_by_cluster_id.iteritems():
        cur_coordinates = [p.GetCoordinates() for p in cur_points]
        median = np.median(np.array(cur_coordinates), axis=0)
        coordinates.append(median)
        cluster_ids.append(cluster_id)
        ratio_in_total.append(len(cur_coordinates) / float(num_total_points))

    mds = manifold.MDS(n_components=2)
    print 'Running mds'
    result = mds.fit_transform(coordinates).tolist()
    print 'Done with mds'

    # Split mds coordinates back to left and right. Exactly one point - median
    # - per cluster.
    left_xs, left_ys, left_colors, left_sizes, left_patches = [], [], [], [], []
    right_xs, right_ys, right_colors, right_sizes, right_patches = (
        [], [], [], [], [])
    for i, (x, y) in enumerate(result):
      if i < len(self._all_left_points_by_cluster_id):
        left_xs.append(x)
        left_ys.append(y)
        left_sizes.append(ratio_in_total[i] * _MDS_PLOT_POINT_SIZE_MULTIPLIER)
        if cluster_ids[i] in colors_by_left_cluster_id:
          left_colors.append(colors_by_left_cluster_id[cluster_ids[i]])
          left_patches.append(
              # This will be used to sort patches by ratio desc.
              (ratio_in_total[i],
               mpatches.Patch(
                     color=colors_by_left_cluster_id[cluster_ids[i]],
                     label='%s: %s%%' % (
                         cluster_ids[i], round(ratio_in_total[i] * 100, 2)))))
        else:
          left_colors.append(
              color_generator.GetKellyColor(color_generator.STRONG_BLUE))
          left_patches.append(
              # This will be used to sort patches by ratio desc.
              (ratio_in_total[i],
               mpatches.Patch(
                     color=color_generator.GetKellyColor(
                         color_generator.STRONG_BLUE),
                     label='%s: %s%%' % (
                         cluster_ids[i], round(ratio_in_total[i] * 100, 2)))))
      else:
        right_xs.append(x)
        right_ys.append(y)
        right_sizes.append(ratio_in_total[i] * _MDS_PLOT_POINT_SIZE_MULTIPLIER)
        if cluster_ids[i] in colors_by_right_cluster_id:
          right_colors.append(colors_by_right_cluster_id[cluster_ids[i]])
          right_patches.append(
              # This will be used to sort patches by ratio desc.
              (ratio_in_total[i],
               mpatches.Patch(
                   color=colors_by_right_cluster_id[cluster_ids[i]],
                   label='%s: %s%%' % (cluster_ids[i],
                                       round(ratio_in_total[i] * 100, 2)))))
        else:
          right_colors.append(
              color_generator.GetKellyColor(color_generator.STRONG_BLUE))
          right_patches.append(
              # This will be used to sort patches by ratio desc.
              (ratio_in_total[i],
               mpatches.Patch(
                   color=color_generator.GetKellyColor(
                       color_generator.STRONG_BLUE),
                   label='%s: %s%%' % (cluster_ids[i],
                                       round(ratio_in_total[i] * 100, 2)))))


    x_lim, y_lim = self._DefinePlotLimits(
        left_xs + right_xs, left_ys + right_ys)
    # Add some between points and right ax border to fit the legend.
    x_lim = [x_lim[0], x_lim[1] + 2]

    # Draw the plot.
    fig = pyplot.figure()
    left_ax = fig.add_subplot(121)

    left_ax.scatter(left_xs, left_ys, c=left_colors, s=left_sizes)
    left_ax.set_xlim(x_lim)
    left_ax.set_ylim(y_lim)
    left_ax.legend(handles=[t[1]
                   for t in reversed(sorted(left_patches, key=lambda t: t[0]))])

    right_ax = fig.add_subplot(122)

    right_ax.scatter(right_xs, right_ys, c=right_colors, s=right_sizes)
    right_ax.set_xlim(x_lim)
    right_ax.set_ylim(y_lim)
    right_ax.legend(
        handles=[t[1]
        for t in reversed(sorted(right_patches, key=lambda t: t[0]))])

    pyplot.suptitle('MDS on cluster medians')

    pyplot.show()

  @staticmethod
  def _DefinePlotLimits(*xs_ys_zs_etc):
    lims = []
    for single_axis_coordinates in xs_ys_zs_etc:
      min_ = min(single_axis_coordinates)
      max_ = max(single_axis_coordinates)
      gap = math.ceil((max_ - min_) * 0.1)
      lims.append((min_ - gap, max_ + gap))
    return lims

  @staticmethod
  def _ShiftCoordinatesToGteZero(coordinates):
    if not coordinates:
      return coordinates

    gte_zero = []
    mins = []
    shifts = []
    for cur_coordinates in coordinates:
      for j, cur_coordinate in enumerate(cur_coordinates):
        if len(mins) > j:
          mins[j] = min((mins[j], cur_coordinate))
        else:
          mins.append(cur_coordinate)

    for min_item in mins:
      shifts.append(0 if min_item >=0 else math.fabs(min_item))

    for cur_coordinates in coordinates:
      shifted_cur_coordinates = []
      for j, cur_coordinate in enumerate(cur_coordinates):
        shifted_cur_coordinates.append(cur_coordinate + shifts[j])
      gte_zero.append(tuple(shifted_cur_coordinates))
    return gte_zero

  def _Mds(self):
    num_left_points = sum(
        len(v) for v in self._all_left_points_by_cluster_id.itervalues())
    num_right_points = sum(
        len(v) for v in self._all_right_points_by_cluster_id.itervalues())
    colors_by_left_cluster_id, colors_by_right_cluster_id = (
        self._GetColorsByClusterId())

    # Parallel arrays:
    points_coordinates = []    # N x M array where M is number of coordinates.
    cluster_ids_for_points = []

    # Mix both sides and run mds.
    for points_by_cluster_id in (self._all_left_points_by_cluster_id,
                                 self._all_right_points_by_cluster_id):
      for cluster_id, cur_points in points_by_cluster_id.iteritems():
        for cur_point in cur_points:
          points_coordinates.append(list(cur_point.GetCoordinates()))
          cluster_ids_for_points.append(cluster_id)

    mds = manifold.MDS(n_components=2)
    print 'Running mds'
    result = mds.fit_transform(points_coordinates).tolist()
    print 'Done with mds'

    # Split mds coordinates back to left and right.
    left_mds_coordinates_per_cluster_id = collections.defaultdict(list)
    right_mds_coordinates_per_cluster_id = collections.defaultdict(list)
    for i, (x, y) in enumerate(result):
      if i < num_left_points:
        left_mds_coordinates_per_cluster_id[cluster_ids_for_points[i]].append(
            (x, y))
      else:
        right_mds_coordinates_per_cluster_id[cluster_ids_for_points[i]].append(
            (x, y))

    # Define data to output at the plot.
    print 'Building subplots for left side'
    left_xs, left_ys, left_colors, left_sizes, left_patches = (
        self._GetMdsSubplotDataForOneSide(
            num_left_points,
            left_mds_coordinates_per_cluster_id,
            colors_by_left_cluster_id))
    print 'Building subplots for right side'
    right_xs, right_ys, right_colors, right_sizes, right_patches = (
        self._GetMdsSubplotDataForOneSide(
            num_right_points,
            right_mds_coordinates_per_cluster_id,
            colors_by_right_cluster_id))

    # Calculate axis limits for both subplots.
    x_lim, y_lim = self._DefinePlotLimits(
        left_xs + right_xs, left_ys + right_ys)

    # Draw the plot.
    fig = pyplot.figure()
    left_ax = fig.add_subplot(121)

    left_ax.scatter(left_xs, left_ys, c=left_colors, s=left_sizes)
    left_ax.set_xlim(x_lim)
    left_ax.set_ylim(y_lim)
    left_ax.legend(handles=left_patches)

    right_ax = fig.add_subplot(122)

    right_ax.scatter(right_xs, right_ys, c=right_colors, s=right_sizes)
    right_ax.set_xlim(x_lim)
    right_ax.set_ylim(y_lim)
    right_ax.legend(handles=right_patches)

    pyplot.suptitle('MDS on original points')

    print 'Show'
    pyplot.show()

  @staticmethod
  def _GetMdsSubplotDataForOneSide(
      num_points, mds_coordinates_by_cluster_id, colors_by_cluster_id):
    xs, ys, colors, sizes, patches_and_ratios = [], [], [], [], []
    for cluster_id, coordinates in mds_coordinates_by_cluster_id.iteritems():
      ratio_in_total = len(coordinates) / float(num_points)
      median_x, median_y = np.median(np.array(coordinates), axis=0)
      xs.append(median_x)
      ys.append(median_y)
      sizes.append(ratio_in_total * _MDS_PLOT_POINT_SIZE_MULTIPLIER)
      if cluster_id in colors_by_cluster_id:
        colors.append(colors_by_cluster_id[cluster_id])
        patches_and_ratios.append(
            # This will be used to sort patches by ratio desc.
            (ratio_in_total,
             mpatches.Patch(
                 color=colors_by_cluster_id[cluster_id],
                 label='%s: %s%%' % (
                     int(cluster_id), round(ratio_in_total * 100, 2)))))

      else:
        colors.append(
            color_generator.GetKellyColor(color_generator.STRONG_BLUE))
        patches_and_ratios.append(
           (ratio_in_total,   # This will be used to sort patches by ratio desc.
            mpatches.Patch(
                color=color_generator.GetKellyColor(
                    color_generator.STRONG_BLUE),
                label='%s: %s%%' % (
                    int(cluster_id), round(ratio_in_total * 100, 2)))))

    return (xs, ys, colors, sizes,
            [t[1] for t in reversed(
                sorted(patches_and_ratios, key=lambda t: t[0]))])

def _CalculateMaxDistanceBetweenBinCollections(
    bin_collection1, bin_collection2):
  """Calculate max distance between means of bins in two collections."""
  max_distance_between_bins = 0

  assert len(bin_collection1.GetBins()) == len(bin_collection2.GetBins())
  if len(bin_collection1.GetBins()) == 1:
    raise ValueError('Can not calculate distance between bin collections.')

  for bin_i in bin_collection1.GetBins():
    for bin_j in bin_collection2.GetBins():
      if bin_i.GetPoints() and bin_j.GetPoints():
        d = _Dist(bin_i.GetFixedMean(), bin_j.GetFixedMean())
        if max_distance_between_bins < d:
          max_distance_between_bins = d
  return max_distance_between_bins


def _MixCollections(bc1, bc2):
  """Mixes two _BinCollection objects."""
  new = _BinsCollection()
  for i, b1 in enumerate(bc1.GetBins()):
    b2 = bc2.GetBin(i)
    new_bin = binner.Bin()
    new_bin.SetFixedMean(b1.GetFixedMean())
    assert all(b1.GetFixedMean() == b2.GetFixedMean())
    for p in b1.GetPoints():
      new_bin.AddPoint(p)
    for p in b2.GetPoints():
      new_bin.AddPoint(p)
    new.AddBin(new_bin)
  return new


def _CalculateDissimilarityBetweenClusters(
      first_cluster_id, first_bin_collection,
      second_cluster_id, second_bin_collection):
  # Sanity check.
  assert len(first_bin_collection.GetBins()) == len(
      second_bin_collection.GetBins())
  num_bins = len(first_bin_collection.GetBins())

  dissimilarity_score = 0
  mixed = _MixCollections(first_bin_collection, second_bin_collection)
  max_dist = _CalculateMaxDistanceBetweenBinCollections(mixed, mixed)

  total_num_iterations = num_bins * num_bins
  remove_prev_line_from_stdout = False
  for i in xrange(num_bins):
    for j in xrange(num_bins):
      cur_iter = i * num_bins + j
      if not cur_iter % 10000:
        if remove_prev_line_from_stdout:
          sys.stdout.write("\033[F")
        print 'Current iteration is %s out of %s' % (
            cur_iter, total_num_iterations)
        remove_prev_line_from_stdout = True
      # Weight of the bin in the first cluster.
      h_i = (len(first_bin_collection.GetBin(i).GetPoints())
             / float(first_bin_collection.GetTotalNumPoints()))

      # Weight of the bin in the first cluster.
      h_j = (len(first_bin_collection.GetBin(j).GetPoints())
             / float(first_bin_collection.GetTotalNumPoints()))

      f_i = (len(second_bin_collection.GetBin(i).GetPoints())
             / float(second_bin_collection.GetTotalNumPoints()))
      f_j = (len(second_bin_collection.GetBin(j).GetPoints())
             / float(second_bin_collection.GetTotalNumPoints()))

      i_mean = mixed.GetBin(i).GetFixedMean()
      j_mean = mixed.GetBin(j).GetFixedMean()
      importance_coef = _Dist(j_mean, i_mean) / max_dist

      dissimilarity_score += (
         math.pow((1 - importance_coef), 2) * (h_i - f_i) * (h_j - f_j))

  if remove_prev_line_from_stdout:
    sys.stdout.write("\033[F")

  return _Dissimilarity(
      first_cluster_id, second_cluster_id, dissimilarity_score)


def _Dist(coordinates1, coordinates2):
  """Euclidean distance between N-dimensional points."""
  if len(coordinates2) == 1:
    return math.abs(coordinates2[0] - coordinates1[0])
  elif len(coordinates2) == 2:
    return math.sqrt(
        math.pow(
            coordinates2[0] - coordinates1[0], 2)
        + math.pow(coordinates2[1] - coordinates1[1], 2)
    )
  else:
    return distance.euclidean(coordinates2, coordinates1)


class _TreePlotter(object):

  def __init__(self, filename):
    self._filename = filename
    self._points_by_cluster_id = {}
    self._total_num_points = 0
    self._bin_collection_by_cluster_id = collections.defaultdict(
        _BinsCollection)
    self._root_cluster_id = None
    self._graph = networkx.DiGraph()
    self._node_sizes = {}

  def Plot(self):
    self._LoadPointsBinAndSeparate()

    for cluster_id in self._points_by_cluster_id.iterkeys():
      self._graph.add_node(str(cluster_id))
      self._node_sizes[str(cluster_id)] = self._GetNodeSize(
            self._bin_collection_by_cluster_id[cluster_id].GetTotalNumPoints())

    self._GenerateRelations(self._points_by_cluster_id.keys())
    self._Draw()

  def _LoadPointsBinAndSeparate(self):
    """Loads points, bins them all and separates bins for each cluster."""
    self._points_by_cluster_id = _LoadPointsByClusterId(self._filename)

    points = []
    for cur_points in self._points_by_cluster_id.itervalues():
      points.extend(cur_points)
      self._total_num_points += len(cur_points)

    good_binner = binner.SplittingInHalfBinner(
        points, min_points_per_bin=_BIN_SIZE)
    bins = good_binner.GetBins()

    for cur_bin in bins:
      bin_by_cluster_id = {}
      for cluster_id in self._points_by_cluster_id.iterkeys():
        bin_by_cluster_id[cluster_id] = binner.Bin()
        bin_by_cluster_id[cluster_id].SetFixedMean(cur_bin.GetFixedMean())

      for cur_point in cur_bin.GetPoints():
        bin_by_cluster_id[cur_point.GetClusterId()].AddPoint(cur_point)

      for cluster_id, cur_bin in bin_by_cluster_id.iteritems():
        self._bin_collection_by_cluster_id[cluster_id].AddBin(cur_bin)

  def _GenerateRelations(self, cluster_ids):
    if not cluster_ids:
      raise ValueError('No cluster ids')
    if len(cluster_ids) == 1:
      return

    dissimilarities = []
    # Calculate distance between each pair of clusters.
    for i, first_cluster_id in enumerate(cluster_ids):
      for j in xrange(i + 1, len(cluster_ids)):
        second_cluster_id = cluster_ids[j]
        print 'Calculating dissimilarity between %s and %s' % (
            first_cluster_id, second_cluster_id)
        dissimilarities.append(
          _CalculateDissimilarityBetweenClusters(
              first_cluster_id, 
              self._bin_collection_by_cluster_id.get(first_cluster_id),
              second_cluster_id,
              self._bin_collection_by_cluster_id.get(second_cluster_id)))

    paired_cluster_ids = set()
    next_level_cluster_ids = set()

    dissimilarities = sorted(
        dissimilarities, key=lambda d: d.dissimilarity_score)
    for diss in dissimilarities:
      if (diss.left_cluster_id in paired_cluster_ids
          and diss.right_cluster_id in paired_cluster_ids):
        continue
      elif diss.left_cluster_id in paired_cluster_ids:
        # If this is the smallest dissimilarity between cluster A and B BUT
        # for cluster A there was smaller dissimilarity with cluster C earlier,
        # make sure that cluster B is not paired with any other cluster at this
        # tree level.
        paired_cluster_ids.add(diss.right_cluster_id)
        next_level_cluster_ids.add(diss.right_cluster_id)
      elif diss.right_cluster_id in paired_cluster_ids:
        paired_cluster_ids.add(diss.left_cluster_id)
        next_level_cluster_ids.add(diss.left_cluster_id)
      else:
        parent_cluster_id = cluster.ClusterId(
            (diss.left_cluster_id, diss.right_cluster_id))
        print 'Pairing clusters %s and %s' % (
            diss.left_cluster_id, diss.right_cluster_id)

        self._graph.add_node(str(parent_cluster_id))

        self._graph.add_edge(str(parent_cluster_id), str(diss.left_cluster_id))
        self._graph.add_edge(str(parent_cluster_id), str(diss.right_cluster_id))

        paired_cluster_ids.add(diss.right_cluster_id)
        paired_cluster_ids.add(diss.left_cluster_id)
        next_level_cluster_ids.add(parent_cluster_id)
        self._bin_collection_by_cluster_id[parent_cluster_id] = _MixCollections(
            self._bin_collection_by_cluster_id.get(diss.left_cluster_id),
            self._bin_collection_by_cluster_id.get(diss.right_cluster_id))
        self._node_sizes[str(parent_cluster_id)] = self._GetNodeSize(
            self._bin_collection_by_cluster_id.get(parent_cluster_id)
            .GetTotalNumPoints())

    # Recursion here can be avoided if needed (performance issues, max recursion
    # depth etc).
    self._GenerateRelations(list(next_level_cluster_ids))

  def _GetNodeSize(self, num_points_for_node):
    return int(float(num_points_for_node) / self._total_num_points * 2000)

  def _Draw(self):
    pyplot.title('Here is our cute tree')
    pos = nx_agraph.graphviz_layout(self._graph, prog='dot')
    networkx.draw(
        self._graph, pos, with_labels=True, arrows=False,
        node_size=[self._node_sizes[n] for n in self._graph.nodes],
        width=3)

    pyplot.show()


def main(unused_argv):
  #_Matcher().MatchAndMds()
  _TreePlotter(_LEFT_FILENAME).Plot()


if __name__ == '__main__':
  main(None)
