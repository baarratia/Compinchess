from IPython.display import SVG, display, Image
import numpy as np

piece_lookup = {
    0 : "K",
    1 : "Q",
    2 : "R",
    3 : "B",
    4 : "N",
    5 : "P",
    6 : "k",
    7 : "q",
    8 : "r",
    9 : "b",
    10 : "n",
    11 : "p",
    12 : "1",
}
def y_to_fens(y, BATCH_SIZE):
  results = []
  for n in range(BATCH_SIZE):
    fen = ""
    for sq in range(64):
      piece_idx = np.argmax(y[sq][n,])
      fen += piece_lookup[piece_idx]
    a = [fen[i:i+8] for i in range(0, len(fen), 8)]
    a = a[::-1]
    fen = "/".join(a)
    for i in range(8,1,-1):
      old_str = "1" * i
      new_str = str(i)
      fen = fen.replace(old_str, new_str)
    results.append(fen)
  return results
