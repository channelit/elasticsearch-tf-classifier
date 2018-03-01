from pysal.cg.shapes import Point
from pysal.cg.shapes import Chain
import pysal
import os
from _config import ConfigMap

training = ConfigMap("Training")
secret = ConfigMap("Secrets")
MAX_LINES = int(training['size'])
NUM_GROUPS = 70
API_KEY = secret['google_maps_api_key']
top = 49.3457868  # north lat
left = -124.7844079  # west long
right = -66.9513812  # east long
bottom = 24.7433195  # south lat


class Trajectory:
    def __init__(self):
        self.json_filepath = os.path.join('/data/logs', 'features.json')
        print("start")

    def get_tree(self, pts):
        tree = pysal.cg.kdtree.KDTree(pts, leafsize=10, distance_metric='Euclidean', radius=6371.0)
        return tree

    def plot_on_bokeh(self, starts, ends, bboxes_start, bboxes_end):
        from bokeh.io import output_file, show
        from bokeh.models import (
            GMapPlot, GMapOptions, ColumnDataSource, GeoJSONDataSource, Circle, Segment, Quad, Range1d, PanTool,
            WheelZoomTool, BoxSelectTool
        )
        from bokeh.resources import INLINE
        import bokeh.io
        bokeh.io.output_notebook(INLINE)

        map_options = GMapOptions(lat=40.7831, lng=-73.9712, map_type="roadmap", zoom=12)

        plot = GMapPlot(x_range=Range1d(), y_range=Range1d(), map_options=map_options)
        plot.title.text = "New York"
        plot.api_key = API_KEY

        source_start = ColumnDataSource(
            data=dict(
                lat=[x[0] for x in starts],
                lon=[y[1] for y in starts],
            )
        )
        source_end = ColumnDataSource(
            data=dict(
                lat=[x[0] for x in ends],
                lon=[y[1] for y in ends],
            )
        )

        source_circles_start = ColumnDataSource(
            data=dict(
                lon=[b.lower + b.height / 2 for b in bboxes_start],
                lat=[b.left + b.width / 2 for b in bboxes_start],
                radius=[max(b.width, b.height) / 2 for b in bboxes_start]
            )
        )

        source_circles_end = ColumnDataSource(
            data=dict(
                lon=[b.lower + b.height / 2 for b in bboxes_end],
                lat=[b.left + b.width / 2 for b in bboxes_end],
                radius=[max(b.width, b.height) / 2 for b in bboxes_end]
            )
        )

        circle = Circle(x="lon", y="lat", size=1, fill_color="blue", fill_alpha=0.8, line_color=None)
        plot.add_glyph(source_start, circle)

        circle = Circle(x="lon", y="lat", size=1, fill_color="green", fill_alpha=0.8, line_color=None)
        plot.add_glyph(source_end, circle)

        circle = Circle(x="lon", y="lat", size=10, fill_color="blue", fill_alpha=0.1, line_color=None)
        plot.add_glyph(source_circles_start, circle)

        circle = Circle(x="lon", y="lat", size=10, fill_color="green", fill_alpha=0.1, line_color=None)
        plot.add_glyph(source_circles_end, circle)

        plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
        output_file("/assets/gmap_plot.html")
        show(plot)

    def points(self):
        file = "/assets/yellow_tripdata_2015-12.csv"
        import csv

        start_pos = []
        end_pos = []
        paths = []

        with open(file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')

            linectr = 0
            for row in readCSV:
                if 0 < linectr < MAX_LINES:
                    if self.is_wihin_range(float(row[6]), float(row[5])) and self.is_wihin_range(float(row[10]),
                                                                                                 float(row[9])):
                        # p_start = Point((float(row[6]),float(row[5])))
                        # p_end = Point((float(row[10]),float(row[9])))
                        p_start = [float(row[6]), float(row[5])]
                        p_end = [float(row[10]), float(row[9])]
                        path = p_start + p_end
                        if not p_start in start_pos:
                            start_pos.append(p_start)
                        if not p_end in end_pos:
                            end_pos.append(p_end)
                        paths.append(path)
                if linectr > MAX_LINES:
                    break
                linectr += 1
        return start_pos, end_pos, paths

    def is_wihin_range(self, lat, lon):
        return bottom <= lat <= top and left <= lon <= right

    def trajectories_knn(self):
        start_pos, end_pos, paths = self.points()
        knn_start = pysal.weights.KNN(start_pos, k=NUM_GROUPS)
        knn_end = pysal.weights.KNN(end_pos, k=NUM_GROUPS)

        start_groups = []
        end_groups = []

        for n in knn_start.neighbors:
            start_group = []
            for i in knn_start.neighbors[n]:
                start_group.append(start_pos[i])
            start_groups.append(start_group)

        for n in knn_end.neighbors:
            end_group = []
            for i in knn_end.neighbors[n]:
                end_group.append(end_pos[i])
            end_groups.append(end_group)

        bboxs_start = []
        bboxs_end = []

        for g in start_groups:
            c = Chain(g)
            bboxs_start.append(c.bounding_box)

        for g in end_groups:
            c = Chain(g)
            bboxs_end.append(c.bounding_box)

        self.plot_on_bokeh(start_pos, end_pos, bboxs_start, bboxs_end)

    def trajectories_dbscan(self):
        import numpy as np
        from sklearn.cluster import DBSCAN
        from sklearn import metrics
        from sklearn.datasets.samples_generator import make_blobs
        from sklearn.preprocessing import StandardScaler
        # from geopy.distance import great_circle
        import pandas as pd

        def centroids(paths):
            db = DBSCAN(eps=0.005, metric='euclidean').fit(paths)
            cluster_labels = db.labels_
            num_clusters = len(set(cluster_labels))
            clusters = [[] for n in range(num_clusters)]
            print('Number of clusters: {}'.format(num_clusters))
            for i, v in enumerate(paths):
                clusters[cluster_labels[i]].append(v)
            return clusters

        start_pos, end_pos, paths = self.points()
        clusters = centroids(paths)  # Array of [start_lat, start_lon, end_lat, end_lon]
        gc = self.createGeometry(clusters)
        self.createJsonFile(gc)

    def createGeometry(self, clusters):
        from geojson import FeatureCollection, Point, LineString, Feature, GeometryCollection
        geometries = [FeatureCollection(
            [Feature(geometry=LineString([(line[1], line[0]), (line[3], line[2])])) for line in cluster]) for cluster in
                      clusters]
        return geometries

    def createJsonFile(self, gc):
        import geojson
        for i, g in enumerate(gc):
            f = open(self.json_filepath.replace('.json', '_' + str(i) + '.json'), 'w')
            f.write(geojson.dumps(g, sort_keys=True))
            f.close()
        pass

    def distance_plot(self):
        from sklearn.metrics.pairwise import euclidean_distances
        from numpy import histogram
        import gc
        start_pos, end_pos, paths = self.points()
        del start_pos, end_pos
        gc.collect()
        distances = euclidean_distances(paths, paths)
        hist, edges = histogram(distances, bins=100, density=False)
        del distances
        gc.collect()
        self.plot_on_bokeh_hist('distance_hist.html', 'Distane', 'Counts', 'Distance Matrix', hist, edges)

    def plot_on_bokeh_hist(self, filename, x_label, y_label, title, hist, edges):
        from bokeh.layouts import gridplot
        from bokeh.plotting import figure, show, output_file

        f = figure(title=title, tools="save", background_fill_color="#FFFFFF")

        f.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#33b5e5", line_color="#4285F4")
        f.xaxis.axis_label = x_label
        f.yaxis.axis_label = y_label

        output_file('/data/logs/' + filename, title=title)
        show(gridplot(f, ncols=2, plot_width=600, plot_height=400, toolbar_location=None))
        pass

    def neighbors_plot(self):
        import gc
        from numpy import histogram
        import numpy as np
        from sklearn.neighbors import radius_neighbors_graph

        start_pos, end_pos, paths = self.points()
        del start_pos, end_pos
        gc.collect()
        neighbors = radius_neighbors_graph(paths, radius=0.15)
        del paths
        gc.collect()
        neighbors = neighbors.toarray()
        x = np.matrix(neighbors)
        x = x.sum(axis=1)
        counts = [d[0, 0] for d in x]
        hist, edges = histogram(counts, bins=10, density=False)
        self.plot_on_bokeh_hist('neighbors_hist.html', '# of Neighbors', '# of Occurrance', 'Neighbors Within Radius',
                                hist, edges)
        pass


if __name__ == '__main__':
    trajectory = Trajectory()
    trajectory.trajectories_dbscan()
    trajectory.distance_plot()
    trajectory.neighbors_plot()
    print('done')
