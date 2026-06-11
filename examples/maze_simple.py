# Simple maze viewer - just opens the maze in PyBullet GUI
import math
import pybullet as p
import pybullet_data
import time

# ---------- Constants ----------
INCH = 0.0254
MAZE_INCH = 140.0
N = 12  # 12x12 cells
MAZE = MAZE_INCH * INCH               # total side length (meters)
CELL = MAZE / N                        # cell size (meters)
HALF = MAZE / 2.0
WALL_T = 0.04                          # wall thickness (meters)
WALL_H = 0.25                          # wall height (meters)

def _x_at(i):
    """World x coordinate of grid line i (0..N)."""
    return -HALF + i * CELL

def _y_at(j):
    """World y coordinate of grid line j (0..N)."""
    return -HALF + j * CELL

def _row_index_top0(row):
    """Map a top-origin row label to 0..N (0=top grid line, N=bottom)."""
    LETTERS = {ch: i for i, ch in enumerate("0ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    if isinstance(row, int):
        return row
    else:
        return LETTERS[row]

def _row_to_world_y_top0(row):
    """World y coordinate for a top-origin row grid line."""
    j_top0 = _row_index_top0(row)    # 0..N (0=top)
    j_bottom0 = N - j_top0                 # convert to bottom-origin
    return _y_at(j_bottom0)

def _add_wall_segment(x, y, length, orientation='h', color=[0.85, 0.3, 0.3, 1]):
    """
    Place a single wall segment centered at (x,y), axis-aligned.
    orientation='h' -> long along X, thin along Y
    orientation='v' -> long along Y, thin along X
    """
    if orientation == 'h':
        hx, hy = length/2, WALL_T/2
    else:  # 'v'
        hx, hy = WALL_T/2, length/2

    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, WALL_H/2])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, WALL_H/2], rgbaColor=color)
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                      basePosition=[x, y, WALL_H/2])

def _add_h_run(row, i0, i1):
    """Add horizontal wall run: row (0 or 'A'..'L'), columns i0..i1 (0..N)."""
    i0, i1 = sorted((int(i0), int(i1)))
    length = (i1 - i0) * CELL
    if length <= 0:
        return
    cx = (_x_at(i0) + _x_at(i1)) / 2.0
    cy = _row_to_world_y_top0(row)
    _add_wall_segment(cx, cy, length, 'h')

def _add_v_run(col_i, r0, r1):
    """Add vertical wall run: column col_i (0..N), rows r0..r1 (0 or 'A'..'L')."""
    j0 = _row_index_top0(r0)
    j1 = _row_index_top0(r1)
    j0, j1 = sorted((j0, j1))
    length = (j1 - j0) * CELL
    if length <= 0:
        return
    cx = _x_at(int(col_i))
    y0 = _row_to_world_y_top0(j0)
    y1 = _row_to_world_y_top0(j1)
    cy = (y0 + y1) / 2.0
    _add_wall_segment(cx, cy, length, 'v')

def build_maze():
    """Build the complete maze including boundaries and interior walls."""
    # Build boundary walls
    _add_h_run(0, 0, N)        # TOP
    _add_h_run("L", 0, N)       # BOTTOM
    _add_v_run(0, 0, "L")       # LEFT
    _add_v_run(N, 0, "L")       # RIGHT
    
    # Build interior maze walls (horizontal bars)
    _add_h_run("B", 0, 10)
    # _add_h_run("A", 3, 5)
    # _add_h_run("A", 6, 8)
    # _add_h_run("B", 5, 7)
    # _add_h_run("C", 1, 3)
    # _add_h_run("C", 6, 7)
    # _add_h_run("C", 9, 10)
    # _add_h_run("D", 6, 7)
    # _add_h_run("D", 9, 11)
    # _add_h_run("E", 4, 7)
    # _add_h_run("G", 3, 11)
    # _add_h_run("H", 1, 3)
    # _add_h_run("H", 5, 8)
    # _add_h_run("I", 1, 3)
    # _add_h_run("I", 5, 6)
    # _add_h_run("I", 9, 11)
    # _add_h_run("J", 4, 7)
    # _add_h_run("J", 8, 9)
    # _add_h_run("K", 5, 6)
    # _add_h_run("K", 7, 8)
    
    # Build interior maze walls (vertical bars)
    # _add_v_run(1, "C", "E")
    # _add_v_run(1, "H", "I")
    # _add_v_run(3, "A", "F")
    # _add_v_run(3, "I", "K")
    # _add_v_run(4, "B", "C")
    # _add_v_run(4, "C", "F")
    # _add_v_run(4, "G", "K")
    # _add_v_run(5, "H", "I")
    # _add_v_run(5, "K", "L")
    # _add_v_run(6, "C", "D")
    # _add_v_run(7, "B", "C")
    # _add_v_run(7, "D", "E")
    # _add_v_run(7, "H", "J")
    # _add_v_run(8, "A", "E")
    # _add_v_run(8, "H", "K")
    # _add_v_run(9, "A", "C")
    # _add_v_run(9, "G", "I")
    # _add_v_run(9, "J", "L")
    # _add_v_run(10, "C", "D")

if __name__ == "__main__":
    # Connect to PyBullet with GUI
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # Set up camera
    p.resetDebugVisualizerCamera(3.3, 0, -80, [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    # Build the maze
    print("Building maze...")
    build_maze()
    print("Maze ready! Close the window to exit.")
    
    # Keep the window open
    while True:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)  # ~240 Hz update rate

