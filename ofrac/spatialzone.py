"""SpatialZone: a rectangular/box-shaped 3-d region, and its command-line spec.

Shared by the `ofracstats_*.py` and `ofrac*.py` command-line tools, so that
every tool parses "start/end/size" zone specifications the same way.
"""

import re
import sys

try:
    from ofrac.p_system.constants import DIR
except ModuleNotFoundError:
    from p_system.constants import DIR


class SpatialZone:            # {{{

   def __init__(self,
         size=None,
         start=None,
         end=None,
         asString=None,
         truncateToZone=False):
      """Make a zone.
         Specify one or two of the { size, start, or end } parameters, or give a
         text string 'asString' that should be parsed to find the appropriate
         info.

         asString must contain one or two (x,y,z) triples. One triple implies
         the size of the zone, which is assumed to begin at (0,0,0).

         Keywords (or abbreviated keyword)
            start | st,
            end | e, or
            size | si
         may preceed triples. A triple without a keyword takes the first corner
         a keyword did not claim, in the order start then end -- so '(a)(b)' is
         start a, end b, and 'st(a)(b)' means the same. A lone triple with no
         keyword at all is the 'size', measured from the origin.


         truncateToZone : bool
            Causes fracture lengths to be calculated/reported for this spatial
            zone, if it is smaller than the whole fracture domain.

      """

      if not size and not start and not end and not asString:
         mx = sys.float_info.max
         mn = sys.float_info.min
         self.c = ( (mx,mn),(mx,mn),(mx,mn) )
         return

      if asString:
         strSave = asString

         e = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
         triple=r"\( *({}), *({}), *({}) *\)".format(e,e,e)
         # one triple, optionally introduced by the keyword naming which corner
         # it is.  Longer keywords lead the alternation so that 'start' is not
         # read as 'st' and 'end' not as 'e'.
         token = re.compile(r"(?:\b(start|st|size|si|end|e)\b[^()]*?)?"+triple)

         WHICH = { 'start':'start', 'st':'start',
                   'size' :'size' , 'si':'size' ,
                   'end'  :'end'  , 'e' :'end'  , }

         named = {}
         unkeyed = []
         for m in token.finditer(asString):
            vals = tuple( map(float, m.groups()[1:4]) )
            if m.group(1):
               named.setdefault(WHICH[m.group(1)], vals)
            else:
               unkeyed.append(vals)

         # A lone triple is the size; otherwise unkeyed triples fill the corners
         # the keywords left free, start before end.  Parsing them positionally
         # is what makes the keyword-free '(a)(b)' spelling work.
         order = ['size'] if len(unkeyed) == 1 and not named else ['start','end']
         for slot, vals in zip((s for s in order if s not in named), unkeyed):
            named[slot] = vals

         start = named.get('start', (0.0,0.0,0.0))
         size = named.get('size')
         end = named.get('end')

         if not size and not end:
            print( "'{}' did not contain enough start, size or end info".format(strSave), file=sys.stderr )
            sys.exit(1)


      if not size and not end:
         raise Exception("Must specify 'size' or 'end' of the zone.")
      elif start and end:
         self.c = ( (start[0], end[0]), (start[1], end[1]), (start[2], end[2]) )
      elif start and size:
         self.c = ( (start[0], start[0]+size[0]), (start[1], start[1]+size[1]), (start[2], start[2]+size[2]) )
      elif size:
         self.c = ( (0.0, size[0]), (0.0, size[1]), (0.0, size[2]) )
      else:
         self.c = ( (0.0, end[0]), (0.0, end[1]), (0.0, end[2]) )

      # normalize data types
      self.c = ( (float(self.c[0][0]), float(self.c[0][1])),
                 (float(self.c[1][0]), float(self.c[1][1])),
                 (float(self.c[2][0]), float(self.c[2][1])),
              )

      for (c1,c2) in self.c:
         if c2 < c1:
            raise Exception("Bad coordinates specified for this zone.")

      self.truncateToZone = truncateToZone

   def expandBoundingBox(self, other):
      self.c = (
         ( min(self.c[0][0], other.c[0][0]), max(self.c[0][1], other.c[0][1]) ),
         ( min(self.c[1][0], other.c[1][0]), max(self.c[1][1], other.c[1][1]) ),
         ( min(self.c[2][0], other.c[2][0]), max(self.c[2][1], other.c[2][1]) )
         )

   # SiZe in a particular direction
   def xSz(self): return self.c[0][1] - self.c[0][0]
   def ySz(self): return self.c[1][1] - self.c[1][0]
   def zSz(self): return self.c[2][1] - self.c[2][0]
   def size(self,d):
       try:
          int(d)
          return self.c[d][1] - self.c[d][0]
       except ValueError:
          return self.c[DIR[d]][1] - self.c[DIR[d]][0]

   def vol(self):
      return self.size(0) * self.size(1) * self.size(2)

   # STart coord in a particular direction
   def xSt(self): return self.c[0][0]
   def ySt(self): return self.c[1][0]
   def zSt(self): return self.c[2][0]
   def st(self,d): return self.c[d][0]

   # ENd coord in a particular direction
   def en(self,d): return self.c[d][1]

   def start(self):
      """Return the starting coordinate as a tuple"""
      return tuple(map(lambda v: v[0], self.c))

   def end(self):
      """Return the ending coordinate as a tuple"""
      return tuple(map(lambda v: v[1], self.c))

   # Range in a particular direction
   def xR(self): return self.c[0]
   def yR(self): return self.c[1]
   def zR(self): return self.c[2]
   def r(self,d): return self.c[d]

   def __str__(self):
      return "x:{} y:{} z:{}".format( self.c[0], self.c[1], self.c[2] )

# }}}
