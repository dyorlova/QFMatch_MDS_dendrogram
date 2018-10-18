VIVID_YELLOW = 'vivid_yellow'
STRONG_PURPLE = 'strong_purple'
VIVID_ORANGE = 'vivid_orange'
VERY_LIGHT_BLUE = 'very_light_blue'
VIVID_RED = 'vivid_red'
GRAYISH_YELLOW = 'grayish_yellow'
MEDIUM_GRAY = 'medium_gray'
VIVID_GREEN = 'vivid_green'
STRONG_PURPLISH_PINK = 'strong_purplish_pink'
STRONG_BLUE = 'strong_blue'
STRONG_YELLOWISH_PINK = 'strong_yellowish_pink'
STRONG_VIOLET = 'strong_violet'
VIVID_ORANGE_YELLOW = 'vivid_orange_yellow'
STRONG_PURPLISH_RED = 'strong_purplish_red'
VIVID_GREENISH_YELLOW = 'vivid_greenish_yellow'
STRONG_REDDISH_BROWN = 'strong_reddish_brown'
VIVID_YELLOWISH_GREEN = 'vivid_yellowish_green'
DEEP_YELLOWISH_BROWN = 'deep_yellowish_brown'
VIVID_REDDISH_ORANGE = 'vivid_reddish_orange'
DARK_OLIVE_GREEN = 'dark_olive_green'


KELLY_COLORS_BY_COLOR_NAME = {
    VIVID_YELLOW: (255, 179, 0),
    STRONG_PURPLE: (128, 62, 117),
    VIVID_ORANGE: (255, 104, 0),
    VERY_LIGHT_BLUE: (166, 189, 215),
    VIVID_RED: (193, 0, 32),
    GRAYISH_YELLOW: (206, 162, 98),
    MEDIUM_GRAY: (129, 112, 102),
    VIVID_GREEN: (0, 125, 52),
    STRONG_PURPLISH_PINK: (246, 118, 142),
    STRONG_BLUE: (0, 83, 138),
    STRONG_YELLOWISH_PINK: (255, 122, 92),
    STRONG_VIOLET: (83, 55, 122),
    VIVID_ORANGE_YELLOW: (255, 142, 0),
    STRONG_PURPLISH_RED: (179, 40, 81),
    VIVID_GREENISH_YELLOW: (244, 200, 0),
    STRONG_REDDISH_BROWN: (127, 24, 13),
    VIVID_YELLOWISH_GREEN: (147, 170, 0),
    DEEP_YELLOWISH_BROWN: (89, 51, 21),
    VIVID_REDDISH_ORANGE: (241, 58, 19),
    DARK_OLIVE_GREEN: (35, 44, 22)
}

KELLY_COLORS = frozenset(KELLY_COLORS_BY_COLOR_NAME.values())


def GetKellyColor(color_name):
  return _NormalizeColor(*KELLY_COLORS_BY_COLOR_NAME[color_name])


def _NormalizeColor(r, g, b):
  return r/255., g/255., b/255.


class ColorGenerator(object):
  """Generates colors for clusters.

  Public methods:
    def GetColor(self, cluster_id): returns color for the cluster.
  """

  def __init__(self, chunk_ids, predefined_colors_by_chunk_id=None, 
               exclude_colors=None):
    self._available_colors = list(KELLY_COLORS)
    if predefined_colors_by_chunk_id:
      if exclude_colors:
        # Checking that predefined colors and colors to exclude do not 
        # intersect.
        assert not set(
            predefined_colors_by_chunk_id.itervalues()) & set(exclude_colors)
      for chunk_id, color in predefined_colors_by_chunk_id.iteritems():
        assert chunk_id in chunk_ids
        assert color in KELLY_COLORS
        if color in self._available_colors:
          # Color can be used N times for predefined chunks.
          self._available_colors.remove(color)
      for color in exclude_colors:
        assert color in KELLY_COLORS
        if color in self._available_colors:
          self._available_colors.remove(color)

      self._color_by_chunk_id = dict(predefined_colors_by_chunk_id)
      self._chunk_ids_to_generate_color_for = set(
          [c_id for c_id in chunk_ids 
           if c_id not in predefined_colors_by_chunk_id])      
    else:
      self._color_by_chunk_id = {}
      self._chunk_ids_to_generate_color_for = set(chunk_ids)

    if len(self._chunk_ids_to_generate_color_for) > len(self._available_colors):
      raise ValueError(
          ('We can not generate different colors for more than %s '
           'chunks yet') % len(KELLY_COLORS))
    self._Load()

  def _Load(self):
    for i, chunk_id in enumerate(sorted(self._chunk_ids_to_generate_color_for)):
      self._color_by_chunk_id[chunk_id] = _NormalizeColor(
          *self._available_colors[i])

  def GetColor(self, chunk_id):
    """Returns color for the chunk."""
    return self._color_by_chunk_id[chunk_id]
      