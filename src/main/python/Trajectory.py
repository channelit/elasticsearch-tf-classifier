from pysal.cg.shapes import Point
import pysal

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
            if 0 < linectr < 100:
                p_start = Point((row[6],row[5]))
                p_end = Point((row[10],row[9]))
                start_pos.append(p_start)
                end_pos.append(p_end)
            if linectr > 100:
                break
            linectr+=1

    wknn3 = pysal.weights.KNN(start_pos)

    tree = trajectory.GenerateTree(start_pos)
    print(start_pos)
    print(end_pos)