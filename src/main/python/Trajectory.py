from pysal.cg.shapes import Point
from pysal.cg.shapes import Chain
import pysal

MAX_LINES = 1000
NUM_GROUPS = 100

class Trajectory:

    def GenerateTree(self, pts):
        tree = pysal.cg.kdtree.KDTree(pts, leafsize=10, distance_metric='Euclidean', radius=6371.0)

        return tree


if __name__ == '__main__':
    trajectory = Trajectory()
    file = "/assets/yellow_tripdata_2015-12.csv"
    import csv

    start_pos = []
    end_pos = []

    with open(file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')

        linectr = 0
        for row in readCSV:
            if 0 < linectr < MAX_LINES:
                p_start = Point((float(row[6]),float(row[5])))
                p_end = Point((float(row[10]),float(row[9])))
                if not p_start in start_pos:
                    start_pos.append(p_start)
                if not p_end in end_pos:
                    end_pos.append(p_end)
            if linectr > MAX_LINES:
                break
            linectr+=1

    # tree = trajectory.GenerateTree(start_pos)

    knn_start = pysal.weights.KNN(start_pos, k = NUM_GROUPS)
    knn_end = pysal.weights.KNN(end_pos, k = NUM_GROUPS)

    start_groups = []
    start_group = []
    end_groups = []

    for n in knn_start.neighbors:
        start_group = []
        for i in knn_start.neighbors[n]:
            start_group.append(start_pos[i])
        start_groups.append(start_group)

    for n in knn_end.neighbors:
        end_group= []
        for i in knn_end.neighbors[n]:
            end_group.append(end_pos[i])
        end_groups.append(end_group)

    for g in start_groups:
        c = Chain(g)
        print(c.bounding_box)
    print('done')