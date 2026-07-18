# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "accelerate==1.8.1",
#   "torch==2.7.1",
#   "transformers==4.53.3",
# ]
# ///
"""Run the frozen Phase 3 natural music-recommendation pilot on HF Jobs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EMBEDDED_BUNDLE_B64_ZLIB = "eNrtfdlyG8mS5Xt/Baweph9GqsFKCWVz24yAhIUiUEWQxDZ3jJYbEglkJlDIBEGg5/77nOMRuYGURJWqdF/KrPsWlcwlwsP9+PElgv/9H6XST+Y+tH3n4dHZRd4m/OmX0k/bpRE5tYfQiPc7w39wndDZGTF++fBY+ekNn9nuNvHG2vgPns0Hgn3kWQ87x1d3nT1ve5G1weuPzx+PYtwT8RXO09bf4Csb3Jb+erHbnJzwwXQWm52Tvd/Y216Mf0Z7P47O3ohPLx2+8b9xGb9YRXggWhrVxgU/06iVjUX9XeO9UzWdRqVh8h/mO2thmlXbaFxYtll911yY7yuOWTGsWtMx3pum4VTrzXfW+/eOfA2vDYzd2t4c8q+u1itW1aovyhcX79+VrZpVWzRM6+Jd9V2l+c4o20bdWlwY761qzS7b9uJdrVkx+YV3VvNdrV43fsKb/1WczNY4+hvDjh7Mi3pxSiKyw8btt68OVtAsm9Xho9lpbuft/gWuxfZ0uJ1Vmsf55Glpd/1H028ejG7nNL8bLsygU7anV0vcvzamw5VZm/tW6C/s1Ufv+lLeGeCdF/NJYzGbjhb2dLTqr3B91T8MVrN48MGKB3ej+8GH+4vBh5uLwd3H/fDDZW24sp6GlU3yjsNseuWb4WgxqzZjc+KX5R3BPJidOt5wtU7uO9ldjmfg/epdevPp8mB2m0d895FjsnpXj3a3uTKrT4t5ePXoBOP9tHrlzwPMbVyY39KejLZ2B9emw4rZG5363uFTv33pWcH4NJ90jrPqcmFNx75VG23Nal2+N+o2fevYWva7lcd5dxxdT2zfDMb8vj9vt8rGdBTNbytLKxxtrdqNO6t2yrPJ0+P82Iohuxj3rYx2C+OoLI1J3Z3dtnZm0KyZXiUbm9eCfK8eLa+1nIcjH//FGDsVY3q1NQPLNSaN9XzawndHkJd8Yz/v+muO4Xoyx3vuXcxBr9M66nfx+3Zri3Hnv1HFuLZ4ruzcttbzyRDz6Xhm996dTW9c3FPmz1jTdb+r5XfbSuc1r473c+iEfleFY8E4uX5bu7eO8d6TETS3HN98Ol+avbG6d/U+WcfYrGL81L9eRLln1zK9HNn4znWlU5tP6qfru1H7+v5qaVXv9TvwTHjl2wG+NWnulc6tD8PgKph/GHnD7mg1rM6eht2xP/wwOA4+XDaG3Zv6bHXzNFy5WBfct3Irg9M6e18wrlvd8crujtfTaqNijpvRbOpDjqJvp6HSEY51Y0xGa+q8VfUv5rcY74d+Zbi6jNoef38JWfor/P4kv2tj3N3OyujexwOvfui7+F6vrO4NUh3SetZ6l7zD7l0tZ9X4NK1BJ2CTZrfTwHtPNuXz4T79FsYbY9x8R8UK7jnWw/XKyv2+eZjWLvX1dTbGSaeOefp2hZgQY/1knuXhBze9Zx74vtkdLczaFfSuz/GZfR/r3H3yYTNuf9052mpMEeZ5C9s5cR7T21RWBdygjZrTVnlaw1pPWxH1VtZufVUxJk/Qt48uvlmFHZWtADKD7Qg+HVu5+bcCs9aHni79fu/KV3YKWwjGkVmz9v0ObPiW+t6Jrtst3+EzXX8PO1zhe0tgRtjvjY6zyXBHezaDG/Wu7riOa2W+Ezh0nKtnEr3He/nOOXU9Am6c+t2mB/17BB65VnUcwY7K1EW8s2Z0x3t8YyPjgP3i/jJs8WTUBi7tqd/FO4HDNtbeua27I7k2xnq7LnDEN28Prl27gpxhh9PlclYbRbDVRzO4d02O67a1he2VDcyt350/WkH5ov+xjjGqtcE6vtd2/D/7vYN7M70CPg3LvD7TP//mtd73O5jrFP4Aa/Bbt8n3ngxgKmQMHGyU8f+r+V3DMv2n/Z3Ikbbfd7mOWKdofrdxnVrraFbnW7M7bk67jQ/wF0c+m1sb2Hm8Sq5T5+aTEe/Fe6kDfiRjwzcGkMVYjfuin7M3YPrarI2hN/xW5QDsxtgq1ImytrGh1pMj8cycVIiRCUaVMcfdtde6Awb6M2JYJtPAmM5l7TkG4GtYWP9kvbCWBtbZzMlgRnzsDYF5Gt+8unsfjOkv9/1u47GP523KqUdMkXEQ1+GfGitgouAWxjQGprm4hjE8bbm+tOXCGE4b9w7vHgseRAk+AwOelM+5q7uLduuW87aCkfiJ2WfX2944d19f6/secX98lGvTAdeP3CGAjfqLMX5f9cv2pCP6Tsya1cZH+J091mU/n4zXvAfre/tsLcLBxbRbf8K877V96zlscrgO3e42sdZD6smJ+DA59Z8GsInBqn+6bl8eB8A3YO7xV7k2UNdOvl3E8s4e+Ex/f+KYrCP0pBMdhx8+8tmKet99jc8OVwNeq6r3WU9y7STfqKv7XPnGr3e8b3AYHnHt9JH3nQbq2pO6NlDX5NlBWV27aeTwMKfL+fkNKgO5977B54cyv0FNXbPk2q8f5Dt1GeNpXcu984B1eYTsM4zsgAfVsB7iK6xjJpfKEngY25PKgn4aOlaFL2/ArlI7nBKXaL8YF+Zdz55dPsIXAO9GC6v7tHTAIcCPyNkq8PX0HdVh+2Xcxz172E3d7g3Eh2j/4mb+X/mb9L6MB8DPjg8YUwXXTg7GZfRgR/A36TuUzmgMEv9H/7ub014n8AnVEXFBPdsGZtQuwYuGnEttBtunjzDp9+mbAn/Vl/cftH0Dc8lnwAnAhSqwFcx1fILuR/gdvjmqzDRHTOWk5n+GdzeKs2YyFzuFzSzNdmFsLrilb7VbeH5EbMc3yso/ffjY+PVD60Qsgi1W5vAV6jq49h2v2xvaJ/xM1az6a4Vl4oO0Dxudro4tj77oehLvzZq9J36ZgQ1/NvCyNSlH37A+mKON+TcYI6xgy9SlV6/PbdA5QM4+MXBWFT6q1wdrUbW3c+j6PNRY2da+uto5mrf8uYO1tZQvSjH27LvBsAI+LbEFeJavxwwOTy58H2O9Q6N3o7+t5A9fDt9lw4+6wnXtifjbl7gDZIlvpvyh8kgeYhBL4QMUp4Ncuk8NrOPSnDQreAb8qBECi1K5LlI7vnRVnJbpEO7Fu5bQIX8tPCmEbGFzVm+8NypNritssXnIfKVap9Q/K98YGZOYsQvmh7WfkrNHjA2I/Vre4IHgLdoeNCca77kuVrBsYH6xMRnQr2DujXBOTA/TuCnTjzM/oOK4J3AA2GL1qmooGQg3y8kRPKZTFpsMGuKfMSfwPPj5SWPX79qIf0aY6wBr78NOqM83scy7x2fsmhX4yneDZ8M2Tngn9H8ezCWuvKFNrWYTxCl/VO5BJ56RC1OPoA8GYjjqe6oTr5M/dOBqS0yAjCHPxtakHLqKO2Lce8rYnFLfgfXd0ZG8IFkP+DPMkZgO/xji/nAc812zyYHzq4LbPFpdrsvNq7BIY09EfUTsSA4JzKt4+hvkKtkYs7US7g1OhHcOGfeKz5jzvsCKCtjTVbEN54Rvwb8sEz62Ioczkng0/MPYo2LPwD7S1807TXC01sastRZfxITPYJEtsfVVSGylvMGtD9eTpfgCq9qBvj0hxmYMOl6fYT/n7cEujxI/dMmpR6dk3eY6/uC9NvjdPICvrg2+ZY2gY5jrbUvFRLXUDrD+wJxAxe7FOR9c4GOc8wHHl/gheOaS2GVJLNEBb23uzd5ax1Z6jlXwUTcZa6uZjTuNVROsxVqOfKc3Whh6/MgzpFiIWGdhBQ2P/AKYuJwFzTTuPZNFyjU1fmyN6hi+sBNh/Y/MSQln7/SDG8xTcXqxFfHz0AXGfVvmI6C7iA9a0Mkh8hCDJB+S+fDu+Vgar8M0+BTaHXQyMpk7qUk+BHGr2E0AfGMcAPuoqG+382sHeUMPCjFnd4j1YexG3Xox/nQ5F87/j2OY5K2O4MO/z6bbi2mV+D4Grkks/xr8khyNFdRdI+hcOB7tAjjY5Rog1wHsAI+i3jOvxvlC18ib/Fhyc8jV6PVO1gH2NlpynfHz0ehBp7tl+t9VggvfYCNl5osk/xT4R8grnsm/GV+3ZL7ncZ2pOF1DYnHacSC2C5tFPDLpEAs4lxp8z4Z8nc+bPV/sB3Nw8QzvXcKmTqJHU8YtHZ0D+KNrNIc+xsgrQddq9pG5AOR+Mg7cG7zSVvge+HjF72DfFeEdKqcAe4YtzeA/LMQX5Cv0Z8w1QBfB7xkHP/f1CZ7B/mAP2gbBbeB7V5jva3iX8vv625AZ5+MyVwtsOyFXCN3i70dYJ8p/cOZPVH4kwa4rruGEPKH1OAuB5zXMG/Odg3Nx3Rnfcs5/FLuQ44unVWLDOILsIR/hm6+T/0f/491tDpsg53n1CbIfblK/gPy1xAH0t90XeABwBc9/Ix6lOedzXyE+Fzpfhx5vHOr0reZ5k0bFqikbgh7Anu4FF7l+OkeCtQEOcc2VPxEuIPxR5+Mw5rjfVbKCDzme2wP4JbjP/R/18yfx7/q7iJvJ+RazGvAVuZBXYhfXKNa4E4Ln+OSO9PlJnhXrshYsSWOQJusAvvLrB3LSQ7J2wLEYXBp1Es1vxF6/Ca8Sv6DjAcgX80IcslQ+QMcfmIfJOJacCxxE2QRlKb4BeJnk+690vIdccUAuV5HaAca5Ya4Kvmap8oWMs5rkzideV1zOV3grtjfcwL6YG8PzfxTDhNeh9lL0199uQ+o9537/WbwOHCGvnsHeiA3Qb8gxSnxMFp/CBiQ2UPH3N9nVfY3rjjxAuywyu2ferDcO7QlzzZWIOAk+Ec5zHA220MB34WNszPnq7Pt98ssy8VfbJ/LUnZXdzvx9ot/f4e99yhv2UpmxzoC5fPMaCAe06c9PCu/VO2krJvTouq1jykl6PYkdUe8AH4MOkE+CdwCXYJMT8mrkS8N1Eud8i82IH0z8BnUa8loneVvE6SfGhvPudim+v93awN9UiV/25Ap+rHEkz5e8DvJYzNHagouNgLGM+EqMETpEzNsTI6A/Hu8vrMGNGu+LOb1q6kdUXu/wDf6m24zm3SbzXE+Yi089NCb2Bv8FLpFbvi6OYQ2TmAaZcPxntlJB5dlfCT9GLDBPsU5shXKS2l8+5oePVRj+qlj/oytrDvnJ2AVThRcLj4KMy+RWkC+4pPCKSuKHzOA9fB7yWvJM8/DH9T7LU87EfuOKxVrvJKm3vE6OjK0SXFQ67PMd5Sw/mI9t7t17b95OOVIu75jxLmL2q3hS4hN0XgTjYNyoeBLiU9+fA8uRv4NPvpH1hv6lMhau2hWum8R/4JGsEcIHhLTN5mO/h5oyfCnm9Ud9cj6PXaYezGujzVR8JfMjjdfmRBCzX8aG8q3UyQr9rfHFvEgRa1ifgb0Htvho1EHBAyXv/7pcoOgPvhGpeGAJ/ugyr8K8QiOxF6mxeTk8YSwQ2MRtHefI2n63vjI2Yc5kWt0unXDzSq4v90aKzxwkdwousRR8Cz5mMW/QZE2Z86mqnDBjHfgF8s/e63BY517JNaQGptckzUeo+FBiblWfO2a2bVWXyHusyZVQpx38CXonGNegT5sH96/0Z7SFVN+wzk+sGafxbD7+sVjXSOsA0G/G3rWR5mfzR+LcN8Q9Ooah72Wtj7zmoyu6A1+IdwUpRvJ6tRkwN2uD/8NevluvwO/RVwO9qoEbTdCn07vavjZHDwyFTSzLHDP000/wD3n5vXHUvHha4Fwn5W8gV2CN8UpePENPDHJ3PvthTMR/0OcKa4ewu6IPoS2GCvtS36S+u4e+cU2hc0PqSUPiecQl896foG/VIbgBeNBkyNxvAP5eebb+n61DNZlPwDhz75C8ImwBeXng3Z4xIuIN5og5Z933oPWvWyHGxOzbsarlV9aapI5UQ2ywS+IJzaOe5Y/zMUTaEyG+siX2ITnMnsQFkq/7E2RJXuIRK/Bd2DBzKcPX5qFUbZG5C/AF1Cigd/OkpkE9AK8ae+xREn+HXLXKkXf2M9bav4F3okfrd7Pmi2+ZSY/HVZKzTDBvSx2FfCR/ovU96Q2gL0t42BrXVoIBuXyEkl3L5n+1bXtWFzGfqqvuyRmSOvu0avMbhR4hXYtgrxJ76JCDE9tmPLtgPkVqNKzjntD7k9WARZdYA0b8u2WPBnDB13XJJfOBwmODOeQ9ZK+O9JRQ/4mNEjvhnb9mtV6tFw1w2flC9eHYi/mE+Tz6suR7ZcXla+jX6zSR22CcOl4wDwLZnNQcO8zjg6cDY4ChzkT1H82Jk5LTSe3NU/mIJni6xvj1WOFUrzgXVVdSc1E1mKbKqzK2qS1VjngCrPgwqA7aLVVLhw7ZgfhFxpS+zqGregi4VB95wLn0K5HHD5k/lDqT1gniPeIflb9Fbhl1Y/QEgK/Y7TRmDw0lS+jG5dPwww1qDazhoO5e6A1i3fFyX+wZS2I4e6FzRMxJFvSCvsuCPemeB+9OdH7YVv4GvAL33EzZU3Wf79VSMajIX3oOfavS9NmPKT1xiIFxjb7kYDH+BQYy74MeS/Bkl98npviZnuV1SHqBtuSqyDujH3JUpq6iprVFfnM5C4fiM4XjjpvIpwPnsabkK/NsfBKrsbfubIyoz3eYP4fPpQ5l709yhdMq6+gV6Cb8hfQdjo72JOuhQ86Z9kPfvkjzboX4TfT2KHOp6px9HquIp4GMK60J5m2oeG9iTx1+l/GTmm/+HlVvREyLdaw0HyEj1CyYH/XBacA/it+Gj79aocdV945qvtApZ318qmeTfb6q32YsfbKs3Sm7nbLXFz031c4CTcqPNq6pmp2yazwLW+mna5zHK3U/cvio9xVxhP0kowblmHIV6K7EnezbGlN/iD1pT1euZ2UQvYBxqv8OdgksXhT6UFXv4vEljAPvWdO/cmxpT9NYYqNKQb8RjwKHFlw7+p2Zx3Fc7gcZzgGL0C8wuaL8kPO5Uj2swu+Uvc7U7/K9xpnOZnzLS/Gs+J4XdWdao+9eKh6X9Pai5gecYr9toR82Nz/WY5asgc1qql8a/BuciTFOmgOKU1uW2CfJ57J22JR+Rein+h37DdjHeCt8IpqDT8y7Kp+b6EGxn2io8g6q/prv85XeN+bzpjW+U/lZfO/IOrMx3frAf91LncmLnJncFb2qzJkvVA/cGD7yiT6WucpnNl3Um8L65HEkHQ90AbXtK6kdz+iXzmrq7AHO6ULqC6A3S+b1iTe6rzAbVzhcMD5VOs7824h+Bf6xJdxL++Z8T24ZHJA1m4WNOgv0Vmpq51iUzB+x/E5673UPVo7TeCn2wd8i9tmnfV0FnMn3H9PmO+D0eU6bjyvYw+JHxfxJTmcTnkCsRw02tWfgG/rdvYTrZDaa83PjJnvYl5SlI7ordRNicYPrTT4l9lsjJl1tHe37X/ARhb4z4swMto7eNoVvyIVaPcUtwCvJHVIcgT5cJHLA+oG3lV/Eu8/rR87/Ir9uJPXkcz+S9EDzu4UY7CrBZvZLUg/pI9L+6tx9ST/XQtU6R1oP6sX1lzz/k0+fpv0s3nvzfB0Qs7D+J/3Z6bjVnodEz+Yi/xbxKslP5/rmC5ie75ln/T+angZV1tGcidSucxg41nU6wT5wbNxL3UH8ytq8ziuhHpPjfD3WyzrUF4WbnabO33RUX/xrsbjTrMLe1Z4M4Tap/9Z5+Bz2IF6hDs6JP6xpKa5l9oMG6yyiN8K7VV+ucKszPEtlluNasE9iJDji9Kwnsn21F3tL7XYsfcRWHut1X/9gjN792iU4Nmp+4J60H+SttsXcV6pXJ+lFV/6B/JE5Eu9FbvJFzNA62dH7BvzUb/8hXMjPifVJHV9ntjBhHUT6Vgr3EquQL09rtMVYXGrXa3I+jSGnJO7KrQHqacLplI0er1rj8dVvN0fZy4CfW7f6Z+qxxKdcyxnjftR4iIMFfwYeDB4ncbLVed8YTobbvl/O7QlRWJbfK6DyNA3WGCVGzLBrjVxag3UI3c9W0Xt2NO4V4oFc/pAcADpX5Fyjxyk5WqcZqrym2GlF+mPaEh/k1jblBhL/I57bSv0QP6ucakvv96qfrcUzXuICNyWXiVwZ/aDEzMm+rKLOKd/Y7yqMuJ7muUMLPp0xHPh7beuf6fUJvDcAF/eZ56FcGWuz1vvSGIlFxdiqAo7FHNGywbwxMPI4kzoWceoz80RPmIq3JB+vZC3xJmIb6SNV+9eEMz3XN67v7+hnQFzPNX9m99Knrsa4fTSe2V1SM1I9ZXqPVXF82rdcT3P7a3p22vf1uecKeJXuZWtl4y3aNnqHryQX5QifTOKtG533yPHQ/L6x3Lefc4GMr+d1SXE8xOvI3xZ5lfiXpxwfC1hfZM8VfGbGO+kPJJdwk/T3od5l+4oHJ3aj43bW3lTfGN4vuZM015X4fsQSK/bJcT/fS/kYvJe1O4lFgUnkkxgP1xs8mvmEZFwFm8vGQZ9/PUn7KvQ+s3qs8s737s14fHtXHrB3ZKX2NOiaPmt+7VRn0ZsE+QOjIP/Mz5zxtgS30NOY4VYb++RuuceoWZM9LroXUvbAfOyM79fND9RPdNbqOOw+P5YLyPexqLMjxvyJLXMfIn3wSfeMlnP9usyNsO+K80Av1/sMK7qFPXLsD/PnxXqFx/oC3yfxJnwNan7LOXN9qNVbNfaoMg+IMeTxs2ibZ+veWOS4BrGV/TO6lpD0xqAHGWssPcbMXTJHG7Iek9pDnPVZDmUvlLoPvlrigT57+o7w2zk8fD4OyQPSrnvP8EL7uLKrfEv/Rf/IWjftu7guoovsec/2pkoM+MK9IXpCJmXpP0b/1Tlu7K/Zm404Av3aig94reu7Il5Qfgfqt9lVdUKdpw0ER6YZHhTiSegVdHQteaJsLYj3zBkXeXWo97F5FeznupKesPSbeu9iOm5/KLWJ2/vzvawZ5yvuq30S/BJcqlYYt62f9a2G7LHY+kmP7wv3ulh/csWyU8hhPruPOrVkP5H08vC9yNEQO2TPIfW2k+YBwYF9lX+T51GnKXJJb4x6JmxrK/uWWbPU+n8t/Zlj2h3sfsx6vOprwrxV7E9s8GPxH13VM4n3N7K+Z/ZxsC4+Zk476esGnxyTp4ieX+fkeZP1peXyIexrln1oHB98iuwXBsdRuKL7OwTfpH6CHLxRlX0KS9X3b7lZjqqet6Eb8tp5bm9zwklkDx/n1yuu12zKepKWsfSvjbFXYaTttRmonsTxQe/zy+mS5IWTvd++6tNPYyj53hmmq15+Xpvm1mMi/XLYh5P0x0pvSgj5CG/hvhj05cWMRbD3o5yX7R2wT+85kfyx4BtiYjOUnna1V1Xlr1P/YLC23q4kPgfx7D37y1i3PbA+BT06lydiMvbHApuAX470hN6omi7z4D3y2SfZE6X3yB4Vt6Y+oCaNfkhL6lvojdU9AuQL3JfBPgm1d0D6BnP8E3xkkuzhVH54erNttt1//OP8bAWecdA/tm7k3ILbS/SOqR5Q6d3KdCS+SbnEjTvK5au4txc+8JH93G13s+K7bOEKQ7/tbrP3ck8r+xxVLVlkmvQ1UW7gL4dkz6nq3Rir9/d0fNnO7SOCv7B07qgv+3g/flK51HqyxzPnPzJd6ZOfAqOxh7KVnEVAudG/muI3bni2gOb63M+m92tjvMDaTxgXcn5j7lMus16qzyVYyl7+UH1H5pPZrN47wD3QMmfYiUXd2hJn0Ysie7Ec4JRgzWErdR9V34Ft3V7+/ik8q8kU5pPlOj95mwvIn/tDxe6TswhUDTTHMxO58NlJ2huj5iA8Od9//sJ5DWK/uT3lH6JP2JvvqvpAZiepDGrSV/W76v8RGaR5Y+g86iLpfuCN1Ivg35jTl73SuRyr2Gyubzbbx673bR82p+vsTIWt/LcrvQd70R18Q9nEfSzxR4o9LfBOxmvSf69qTOwXvW3psxZaWew3HYhda3xN9RE+IcmNMQZIebucVaF895L/thFPCCbKPsQxx57GGtYpEjm0VzfJ/veY+xpY27dZo8HeuuupxIOonVcyGXaZe6/HOo/lqvFIXJjVEbpJboM6ptevNtyqfRf1T21f7dGGrALmZrUdgh+RkwmeMt/BPgL2uclaqp7GTk3OHxF/WHdvp/RDQ54RwfrIRnog2A8oNRDEDu4W9SHIDnxS9avPkfPlHvdkn016zobYQ2ENZS9gZ0X8QDwt+3bs7vt0/6CyZdxfa3Hv/ye9RwTzfpI4HPI6qRyxu9c49RFjRd1T7UcA5p2AV/eG3g+kzkNhj2nnlMXQLS+xTXn/R5s8Us4ayPnT+KWYfjZFLQUYYul97rDFPbC9rPCLe66fHpOavupB67A+r+dXqGvCx8ieFrkP9YUT7ZqySPIjxEPNa7O48iz/qzCQ8uaehiyOoh4Y5IMay1J8wF4V2CPxDny3ueU6XLubT/3jwL3THAP5d8Tegvsix3xdI1sjP8H/g/RCEZM6clbJ4/0UOLMaxMOP/fgGecn55JJyrapaW519g0fdA1mRvWTgVu1QvQv6qTCmy/X6yB5g9HIN4oT/5HIyObmjJkLbqar69zV7yhHLm+z91PuE1ZkT6JHoHrh2qAWDw7EnAnlG5rK0vqb7DFUNwkr36ege21Xa66/m3pN6geQYwDe0z2vL3vumXk+ue1ZvUvul9JktwiWRo52UU13O5b9c1QdxQ7uGXKW/MWdbFenB477S3N5yd6Z7q4VXHivScwxMbiCW2lxPdd9mfp8x8Y5xXE96mT5l+x8PX9wP+FLv50t7ya7bz/ZMfcJaRDj3IN/LHCX7RaTnLBglZ5+smKcw1F7CgbyTa9Tj+xp6v1naHyQ9ke0w6/UmluGchnrO7vU5CKJn5HM+zwaADNK9/0nvj+C97K3hWQSXrq3Pnkh6SmgzY/FbaRyR1okE88JREr/q/c/c/ygxmq/wUutf1w9k/wHk6cjZH0ufvlT1fYgdf9K1yP1ZTBIU4xLV78pexjRHyn7X7hP2L/r7NAbr8cygTqRyU8pnKBtWPR9Oj/Pnfi69DxL7P4iXUp8XnjLa6nqzK/3aHzaf2l654B/ZayO8D1ycspa+nNvUp4tvznIbc91HwrmMj78q+X+6vi3kQ3zdS413+PDz9+6L+TrGGqzZ5s+5yXErbcsF30U+8gLOwx63V+B1vqH3j+jYQLiu7FnQa49ztITbop55yp0V9UKPi5atygMDuxRWtcMv9+pk+yTTHqIEF1IdkZiiqs5lsdtJnzn7vocb9lvhG4hR7A3XE2ujz7iS8xSir/XdvIRL+X4lLSc1Xz2usziKfcwyV9jeXu+zT+xQ5jeTHN9NxgN6Q+6VyuPaXuspcwVe0k+WHxPPcpJat8TSOKfkA3r9sjM6yClbmT5JL1KqS8V86/2nWS5G0vW/k9IdeS7LFaUxTSXZu5L2Uidzy/Khcj6T5EM17vvtoFgLJ6bcsTeWvEXlLJI6MOd7EFx/MU+N85Ok7q50QefKVN9WN6nN38ieTR1PfFKYmfLwJG+l6lKyT/gs5yh18UO2Jm3aAnSyl56xlj8fQrg8dVvqXe2Kro+1WO/i+VT71I9Pbvh81utzVD6AvVvXEx3npXtqdEw4uTrp85W4j5z+PGY8jTrLNuGU2F9ZlpgwEN7rm2GGPWku7baVq50CCz2crtOBP2QuXHJE9lbOf1D5AWBlVpvl2gvf1+uc8QjpIymrNUYsLBzkJkpjUPLgdsY/sziuIO9I86Ekl5fGIjo+SGWu9zQqe4OeqfcrP8IxqLr2+llOJtED5t0oX13TU32A5N/VIZ69UnHQsVXM4XQvJa88Y67jNn92XyUdj/QlUqZr8jCeHddK6uIZ55W935VkjzQxbK1i66HirDneeT1R/ijhg2e5EeWLuXcBYxduJOdipbEfe4nL6puSZ1J9QMkew8QXsV81gG/mWSzCW1nnGp9+VWf7gUO0GteXW9aNGfczN4FeqHpUyG33ijHbdTs9F9DNYjuVA1b5oFyOQc5i5L5t6WX5hHfwzBf4taXsI8vpEOwFeWHprzy4WZ47qQMnPfuyl5u8P5E98brIMVk3AfewJA+yzsce7tdiD30eEGNyyV2mcb6Kvz+p/pKD+7y/hHkF6S9J/Msd90WrvXF6D9jlJlnTLKaRmED3uOZqMAnuKu7FPcfqTBusXY1rwzOm0n2rt/lzUbC392Zbk/z4ROUBWTtM9CLvi1Ls0Riey2eytghsUXEze6vttsrlqpxikl9H7iOUmkOkYwc5A4nnC1rewX051hzivADi6+frwtftVr6PQuHC5VbiPJwnwF4x6esEL1R9y8xB5POg6hxOycEnuaV8jZfjgjwPgo/KB6c97HgXdTDpQWaO9ih7xgKpAem8X6Kzqi9dc4s073Fde7FW+ok1LsnneQU/D97EnOITeXRZ8i5eq9jvclvIr2e9bUeNAcy7TZAX43pNZsKtr7OeZLEFFdsyz5n2BZ7S/S9SF1S4mOzreZZ7Vs9XiEviUz6e1z+x/xq+g3lVxSWkT1pqFODryH/ymwX+LeM7y1nndfCg8/3gIfpMNaVjctYVY6LnuKZzij3Nq16shXLvA89OUDLN6gGq3sIxJjlVFcek6yhzmalck9REed7dme89rzO71CnGMPp5ncMjf5J6yKev5MQe5+n+7gRT4QfY4127OsuZqhhZ55ae5VTJw+41pqL2pWwPPe261ojzXbE2tT557Ys1xPS8Iu5lER4s1xPbi2eq1vXI82nbcoYX9ygqzPl8PfEgOQzmvtI9lkktROpNshePeQ3yJq6F6L/Bfci1QYKzHegH9zCovryEQ6A3Dnvshd+c1xAVtqGGUJU9A1cj4Ya5frfblo6bJa/kGarWXU3O2ymcfehun2zhruIXda6Fe8pY70ue4fmpjURWrGHKnJJck8U9QKpesZFY+oPsNUFOADmVKeOErD5n1SQmismj2dvAWEnOOQE2qv3OiIV7rBnOZS/S7b34CanP5/YP6xqa8rVY9085n13oSYKOd2yJaT9Kjk3lOJFzareK9RPhxOwjpj3L2Gpy7quqBek4UHAcnHp80tiKXJk+5yiXo9Q8SPUbVf2QdYjrxP9IXDiUc0FUblfOycJ5lq2yPk9OzlmmXzJUTpp7ozb8HnIjWzP18yP2BGFdwcXdzT+yw7eDje3IIeiev4nTw8WDbfwQOyguGrGjDz1/fqw5H02ORL85OOH/4v/U3tZbb1s4ED1/z8559JLT1pvlC3Nh1826bb1bVOqOU69Vy+Vm3X5v1isX7+uGuVi8b9Sa6nmck+5YsWM/hPvgwTeOOLUdL6ldyC+tTRg7TzGv/B+ppakTw7PfJEN3nO2DtY8fomMYL7ebrS69qbkuPF8k4HtrJyoFjr+xPaskt77FvW9KOJrcPhiPzpuS42Mwu02I38tvjNDGD76HM9jtkrmPS/twH+0Nv4TX2nuLp7dnn7L2u50Txg+h4xQE6oVuaefgofSTJQ64hAFHJRwJH5TipVOqNN+XI/y02+zdpVypliu4clhi+CXjcePZfBF/YexiL4qj/yyZThS/XYeoLZYi/NJ3omw0uVPu1XpzSK2959slo7TwHp238c6w1iU5674ERThimnEJJ9PLN3bO73u8HZ/wN4efS/eRI3OAiJ5wG0diGbHhbzAiviYSUfFB58mwYtxrbXYyYDXYn0u/7ZyFsythjFFpYz56m31UikTeGCKn6YSFDxsRXsrRpHr5c6mDfzqGtSx50N03pcMO/1Uf9I+lTeiUfA//4/FFnjwc/FIa/ly682Lf+aX0v2P+979K/690KWPCFTU4Xho5Bo6hxyW+JlpudjFGBx0LLee//hn+M7zGfZRoSSvUL6W/Rp3+GbaVGiWC+KX079AjTvk3rRO//DOsJEL8SdTrX2++ZI07x8VE8IcSLC/GHz/4ojVeGVsjdKBcvFcJydxsXA/CW+yJKEpoB2MX4AfRt9eYnlEisL0NPXcZy6tFWpETlw5evCzBNvbUlDMVNp3jRmtxsIEKepjVLhQbMnxo2MIIRGs3ofu3of1oQ/suTXnBqv56FfkeG9ri5Q/bfbh+gOS+YkO89y3vfUMdUKIJnUNJYZCoixNSrZbHeBlAB43dzghdJ4A8os/bUEGBBHe0DOKlAYltwhAqlX285O692NhFSnwYiJInBEHtEQW1dhu80vYW0E+uhe1Yhv23z/rhpvRdCvOCKf0wTfkeizIC06N6B17oBXROShW+YFf6iTcl/UjenYsGvinZ+JcW2RZ6y6F6YRTv9hQVnrCdR+jilv/6vKGJxHxM/y1XP/lsCXqTfFiLk5q12Fj7COzBl7WFsN5kPl3MB4PjHxZ6EjOC8AMQf/n5bzv70Xb25yrQC4b3ozTne+wOf43L2blgo6H9sPS2y8324SUG98z6cs+V8NzbJVEqMhCyOm9NgxPJXoOZmBGnSp004kgJ1BZlxA2PTmkBTY2+YoKGb+6DRGKCTYET4G94meDN+W+FRpAw6HNVLsGqlg6txAihdNA83qUednZ/G+APNsC/QIc+Z4V/ufJ8l+/zNUt9xF+dC80HRKi+YW6UBn7JA2bPlUb/o1UIaKPNHgoH/gucwUI/bizD10JLotrCZ75mfbmb8TXilcO/t+fYb+XVqWghISenyxbugKLZz6WJYcXMJv1tdD/Y6/2ZSvM5c/urtOV7rIyp0F34sDJOpwfy5wC2/6VEpNxe4u0AJThr8AP9FESCvwaJOWDx7JIX4FnkV404jXJtJzbwMltTdCyM9Robi5G/RaC78Hag4hZ+3u+OMoBMHlGJrzWPKfKVTIN/TxP0e4dhEdk2e66Gf6T0QBTwNy9D29jZ0d9k89+aiPwL1Olz1vcD9eh7DHKx8dcPSVzp7B6SMkH0BavkM28kC+rs3mbPKv8e5bDLsKA48VlwrMQZ8pIAk38Eq34d78yyuBE4/064vwS9OZouKqdU5T8jlYBKNdhz5e/TOhJI/219P9j6/gq1+TLV/FP15XusDH9fEoU7KMgrUiooBad84E3J9b3YWqrJsw6483TUm2MM8uavcseiDgr9Joj4zlMCaXmKgaUwdw4LNXiXKE3kbA3KPtTvSFiByFeGQI6RwyXyeOdvK/vBVvZH1eezLPLfozffY20Rg1n5lni2B8Vov+TQPF/NAns7aOzJ4xGTT44RqPTvZmcthSL4WaXSiIEYOLZ5J8S9UPN60QoV85Y7oVDRxvIMUm2RKkcRURYcRfSZMBhLSDXCuucjYd5DCW4iXCHT9yxVbilZS4Px+t+JlR/v7v48pfqcbf77tOl7zJNWHj7k8y1fd4rXfCYfNhdCZmuPrK6hKjW7jbVWYkxq7CWplFBRvmadtyifovHgLSf9GxZ670LYzlsfJGRvuM5ZgdP1N2b+7UifxZq2a13XSKfw5+9w70db4J+kNJ+zvr9WW74ry4JY1X+I9iYUbfca88IFF+FvRGItDwOjNhssoxQk9RVV9LSWEJ7U8dOIeOkYj0elrF8xsKwXR8qeKu5FqTMCBbdXBsNk/bZk8Ep57U0p3EiZFH0M0AEvAFfYYQykCraEzX+b1w8vkf9ZSvM5A/srteUr5oX//b/SXhlBfbPeymqlorSsWq0lP7xLfmjqH2rJr2rN9DWZSuJdymp/UtrzAB/8kOggftnQJ6Kw85SPIPzE1fLP75LrLA7KlWZ6dsoTrOzwEG/WyGbhV/X6+7S1VVF1bQvOwnuiLWSzLfaX5gwHIXjI4byrqhZW48kL0H0KKo+S0IMi18k9tYuyuokF1c/eVC6n/apeuFdfccFXMoG8+DwC91C6W/ULcvftQw/q8kCDY3SPLN4D19wHXXkQhX+wfCQSVLvse/2wIVb34C0exL4fHI/E50G/U8IHith0YEPOA0RjhOyceoDS4EtbWMfD0XN8+2EBPebi6PfudIOToOpH6ZjKDayUDIwV7w2xYQvzeZJ8BxQTEixJ83FJTRfQoVLyelgl+SRfEokx6gasHRFBkMZkorBWBnXUSOLyPniSUqVeL7+Vt2ZwAgDb/py1P1u+4QUPJhmqsfOcTN1/uiOY4P8MqXpzgm/BVYljUby3j2JtlhHS2mC2Knsj4GfgI7qkV1rC8PeWpzrAcNV5hJQALj8nomtrOE1kBJdNKMJ35cWPns3bFQiorjL0U5fWbMcxhP2mo0tf+dtuYzrM9mxsw/R8tsPp1wEqVo4qVhpohIPGR0KT089sFnzO2Rbed4fPikq9daQ/NfJkMh5R0HmrPbd0KOCfqawy43sj31YcwD/KXVix7T5OvzDcqEY5rhRfvduHLM6K0+CkES6f4CeW0AM+SulihjQgdEVEkRPl5y5yt52tvzky3ieq76E/WCKfjbTijnZOjMV+JEj3P0RvRLmS/LcITtofGJrr5cGtnq3FTGD7j3/9f4gWdog="

INLINE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?title"
    r"(?:\*\*)?\s*:\s*(?P<title>.*?)\s*\|\s*(?:\*\*)?artist"
    r"(?:\*\*)?\s*:\s*(?P<artist>.*?)\s*\|\s*(?:\*\*)?reason"
    r"(?:\*\*)?\s*:\s*(?P<reason>.*?)"
    r"(?=\n\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?title"
    r"(?:\*\*)?\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
FIELD_PATTERN = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\d+[.)]\s*)?(?:\*\*)?"
    r"(?P<field>title|artist|reason)(?:\*\*)?\s*:\s*(?P<value>.*?)\s*$",
    re.IGNORECASE,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_bundle(path: Path | None = None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if (
        EMBEDDED_BUNDLE_B64_ZLIB
        == "__PHASE3_NATURAL_GENERATION_BUNDLE_B64_ZLIB__"
    ):
        raise ValueError("pass --bundle or embed the Phase 3 generation bundle")
    payload = zlib.decompress(base64.b64decode(EMBEDDED_BUNDLE_B64_ZLIB))
    return json.loads(payload.decode("utf-8"))


def validate_protocol_payloads(bundle: dict[str, Any]) -> None:
    for key in ("json", "markdown"):
        payload = base64.b64decode(bundle["protocol_payloads_b64"][key])
        if sha256_bytes(payload) != bundle["protocol_hashes"][f"{key}_sha256"]:
            raise ValueError(f"embedded protocol {key} hash mismatch")


def clean_field_value(value: str) -> str:
    return value.strip().strip("| ").strip('"\'').strip()


def parse_playlist(text: str, maximum_tracks: int = 5) -> list[dict[str, str]]:
    inline = [
        {
            field: clean_field_value(match.group(field))
            for field in ("title", "artist", "reason")
        }
        for match in INLINE_PATTERN.finditer(text)
    ]
    inline = [row for row in inline if all(row.values())]
    if inline:
        return inline[:maximum_tracks]

    rows: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        match = FIELD_PATTERN.match(line)
        if not match:
            continue
        field = match.group("field").lower()
        value = clean_field_value(match.group("value"))
        if field == "title" and current:
            if all(current.get(name) for name in ("title", "artist", "reason")):
                rows.append(current)
            current = {}
        current[field] = value
    if all(current.get(name) for name in ("title", "artist", "reason")):
        rows.append(current)
    return rows[:maximum_tracks]


def encode_artifact_chunks(value: Any, maximum_chars: int = 7500) -> list[str]:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    encoded = base64.b64encode(zlib.compress(payload, level=9)).decode("ascii")
    return [
        encoded[start : start + maximum_chars]
        for start in range(0, len(encoded), maximum_chars)
    ]


def summarize_rows(
    rows: list[dict[str, Any]],
    bundle: dict[str, Any],
    architecture_gate: bool,
) -> dict[str, Any]:
    expected = int(bundle["expected_generation_count"])
    parsed_count = sum(int(row["parsed_track_count"]) for row in rows)
    nonempty_count = sum(bool(row["completion"].strip()) for row in rows)
    generation_ids = {row["generation_id"] for row in rows}
    context_counts = Counter()
    for row in rows:
        context_counts[row["context_id"]] += int(row["parsed_track_count"])
    return {
        "generation_count": len(rows),
        "expected_generation_count": expected,
        "unique_generation_count": len(generation_ids),
        "nonempty_completion_count": nonempty_count,
        "parsed_track_count": parsed_count,
        "minimum_parsed_track_count": int(bundle["minimum_parsed_track_count"]),
        "maximum_parsed_track_count": int(bundle["maximum_parsed_track_count"]),
        "parsed_tracks_by_context": dict(sorted(context_counts.items())),
        "architecture_gate": architecture_gate,
        "technical_gate": (
            architecture_gate
            and len(rows) == expected
            and len(generation_ids) == expected
            and nonempty_count == expected
            and parsed_count >= int(bundle["minimum_parsed_track_count"])
            and parsed_count <= int(bundle["maximum_parsed_track_count"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = datetime.now(timezone.utc)
    bundle = load_bundle(args.bundle)
    validate_protocol_payloads(bundle)
    if bundle["mode"] != "pilot":
        raise ValueError("this frozen runner accepts only the Phase 3 pilot")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3 natural generation requires a CUDA GPU")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(
        bundle["model_id"],
        revision=bundle["model_revision"],
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        bundle["model_id"],
        revision=bundle["model_revision"],
        torch_dtype=torch.float16,
        trust_remote_code=False,
    ).to(device)
    model.eval()
    architecture_gate = (
        hasattr(model, "model")
        and hasattr(model.model, "layers")
        and len(model.model.layers) == int(bundle["expected_num_layers"])
    )
    if not architecture_gate:
        raise ValueError("loaded model does not match the frozen architecture")

    rows: list[dict[str, Any]] = []
    for context in bundle["contexts"]:
        prompt = context["generation_prompt"].rstrip()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for seed in bundle["seeds"]:
            torch.manual_seed(int(seed))
            torch.cuda.manual_seed_all(int(seed))
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=float(bundle["generation"]["temperature"]),
                    top_p=float(bundle["generation"]["top_p"]),
                    max_new_tokens=int(bundle["generation"]["max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            completion = tokenizer.decode(
                completion_ids, skip_special_tokens=True
            )
            parsed_tracks = parse_playlist(
                bundle["completion_prefix"] + completion,
                maximum_tracks=int(bundle["generation"]["tracks_per_playlist"]),
            )
            rows.append(
                {
                    "generation_id": f"{context['context_id']}__seed{seed}",
                    "context_id": context["context_id"],
                    "seed": int(seed),
                    "completion": completion,
                    "parsed_track_count": len(parsed_tracks),
                    "parsed_tracks": parsed_tracks,
                }
            )

    summary = summarize_rows(rows, bundle, architecture_gate)
    summary.update(
        {
            "run_id": started.strftime("%Y%m%dT%H%M%SZ")
            + "_phase3_natural_pilot",
            "protocol_id": bundle["protocol_id"],
            "protocol_hashes": bundle["protocol_hashes"],
            "mode": bundle["mode"],
            "prompt_template_id": bundle["prompt_template_id"],
            "model_id": bundle["model_id"],
            "model_revision": bundle["model_revision"],
            "expected_num_layers": bundle["expected_num_layers"],
            "observed_num_layers": len(model.model.layers),
            "bundle_canonical_sha256": sha256_bytes(
                json.dumps(
                    bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ),
            "submitted_script_sha256": None,
            "script_execution_mode": "inline_python_c_no_file",
            "maximum_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    artifact = {
        "rows": rows,
        "summary": summary,
        "claim_boundaries": bundle["claim_boundaries"],
        "continuation_gate": bundle["continuation_gate"],
    }
    for row in rows:
        print(
            "PHASE3_GENERATION_ROW_JSON="
            + json.dumps(row, ensure_ascii=True, separators=(",", ":"))
        )
    chunks = encode_artifact_chunks(artifact)
    for index, data in enumerate(chunks):
        print(
            "PHASE3_GENERATION_ARTIFACT_CHUNK_JSON="
            + json.dumps(
                {"index": index, "total": len(chunks), "data": data},
                separators=(",", ":"),
            )
        )
    print(
        "PHASE3_GENERATION_SUMMARY_JSON="
        + json.dumps(summary, ensure_ascii=True, separators=(",", ":"))
    )
    return 0 if summary["technical_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
