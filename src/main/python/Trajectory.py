from pysal.cg.shapes import Point
from pysal.cg.shapes import Chain
import scipy
import pysal
import os
from _config import ConfigMap
import numpy as np
import numpy.ma as ma
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
# from geopy.distance import great_circle
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.spatial import distance

training = ConfigMap("Training")
eps = float(training['eps'])
grpsize = int(training['grpsize'])
MAX_LINES = int(training['size'])
secret = ConfigMap("Secrets")
matric='euclidean'
NUM_GROUPS = 70
API_KEY = secret['google_maps_api_key']
top = 49.3457868  # north lat
left = -124.7844079  # west long
right = -66.9513812  # east long
bottom = 24.7433195  # south lat


class Trajectory:
    def __init__(self):
        self.json_folder = "/data/logs"
        if not os.path.exists(self.json_folder):
            os.makedirs(self.json_folder)
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
        file = "/large/yellow_tripdata_2015-12.csv"
        import csv

        start_pos = []
        end_pos = []
        paths = []

        with open(file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')

            linectr = 0
            for row in readCSV:
                if 0 < linectr < MAX_LINES:
                    if len(row) > 10:
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
                if linectr % 100 == 0:
                    print("processed {}".format(linectr))
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

    def trajectories_dbscan(self, dist_type):
        def centroids(paths):
            def euclidean_dist():
                return euclidean_distances(paths)
            def cosine_dist():
                return cosine_distances(paths)
            def custom_dist():
                return self.custom_dist(paths)
            distance_calc = {
                "euclidean": euclidean_dist,
                "cosine": cosine_dist,
                "custom": custom_dist
            }
            if dist_type=="default":
                db = DBSCAN(metric='euclidean', eps=eps, min_samples=grpsize).fit(paths)
            else:
                distances = distance_calc.get(dist_type)()
                db = DBSCAN(metric='precomputed', eps=eps, min_samples=grpsize).fit(distances)
            cluster_labels = db.labels_
            num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            unique_labels = set(cluster_labels)
            clusters = [[] for n in range(num_clusters)]
            print('Number of clusters: {}'.format(num_clusters))
            for i, v in enumerate(paths):
                if cluster_labels[i] != -1:
                    clusters[cluster_labels[i]].append(v)
            return clusters

        start_pos, end_pos, paths = self.points()
        clusters = centroids(paths)  # Array of [start_lat, start_lon, end_lat, end_lon]
        raw, clustered = self.createGeometry(clusters)
        self.createJsonFile(raw, "raw_")
        self.createJsonFile(clustered, "c_")

    def trajectories_hdbscan(self, min_cluster_size):
        def centroids(paths):
            # distances = euclidean_distances(paths)
            # distances = cdist(paths, paths, 'euclidean')
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            cluster_labels = clusterer.fit_predict(paths)
            num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            unique_labels = set(cluster_labels)
            clusters = [[] for n in range(num_clusters)]
            print('Number of clusters: {}'.format(num_clusters))
            for i, v in enumerate(paths):
                if cluster_labels[i] != -1:
                    clusters[cluster_labels[i]].append(v)
            return clusters

        start_pos, end_pos, paths = self.points()
        clusters = centroids(paths)  # Array of [start_lat, start_lon, end_lat, end_lon]
        gc = self.createGeometry(clusters)
        self.createJsonFile(gc)

    def createGeometry(self, clusters):
        from geojson import FeatureCollection, Point, LineString, Feature, GeometryCollection, Polygon
        from shapely.geometry import MultiPoint
        total_clusters = len(clusters)
        raw = [FeatureCollection([Feature(geometry=LineString([(line[1], line[0]), (line[3], line[2])]), properties={"index":i, "total_clusters":total_clusters}) for line in cluster]) for i, cluster in enumerate(clusters)]
        def bbox(bounds):
            # l=0, b=1, r=2 t=3 (l,t), (l,b), (r, b), (r, t) (0,3), (0,1), (2,1), (2,3)
            return Polygon([[(bounds[0],bounds[3]),(bounds[0],bounds[1]),(bounds[2],bounds[1]),(bounds[2],bounds[3]),(bounds[0],bounds[3])]])
        def blen(bounds):
            # l=0, b=1, r=2 t=3 (l,t), (l,b), (r, b), (r, t) (0,3), (0,1), (2,1), (2,3)
            return max(abs(bounds[0]-bounds[2]), abs(bounds[1]-bounds[3]))
        clustered = []
        for i, cluster in enumerate(clusters):
            starts = MultiPoint([[line[1], line[0]] for line in cluster])
            ends = MultiPoint([[line[3], line[2]] for line in cluster])
            feature = Feature(geometry=LineString([(starts.centroid.coords[:][0]), (ends.centroid.coords[:][0])]), properties={"size":len(starts), "index":i, "total_clusters":total_clusters})
            # start_bounds = Feature(geometry=LineString([(starts.centroid.coords[:][0]), (starts.centroid.coords[:][0])]), properties={"radius":blen(starts.bounds), "index":i, "total_clusters":total_clusters})
            # end_bounds = Feature(geometry=LineString([(ends.centroid.coords[:][0]), (ends.centroid.coords[:][0])]), properties={"radius":blen(ends.bounds), "index":i, "total_clusters":total_clusters})
            start_bounds = Feature(geometry=bbox(starts.bounds), properties={"size":len(starts), "index":i, "total_clusters":total_clusters})
            end_bounds = Feature(geometry=bbox(ends.bounds), properties={"size":len(starts), "index":i, "total_clusters":total_clusters})
            clustered.append(FeatureCollection([feature, start_bounds, end_bounds]))
        return raw, clustered

    def createJsonFile(self, array_of_featurecollection, prefix):
        import geojson
        json_filepath = os.path.join(self.json_folder, prefix + 'features.json')
        for i, g in enumerate(array_of_featurecollection):
            f = open(json_filepath.replace('.json', '_' + str(i) + '.json'), 'w')
            f.write(geojson.dumps(g, sort_keys=True))
            f.close()
        pass

    def distance_plot(self):
        from numpy import histogram
        import gc
        start_pos, end_pos, paths = self.points()
        del start_pos, end_pos
        gc.collect()
        hist, edges = histogram(paths, bins=100, density=False)
        self.plot_on_bokeh_hist('distance_hist.html', 'Distance', 'Counts', 'Distance Matrix', hist, edges)
        # distances = euclidean_distances(paths)
        distances = self.custom_dist(paths)
        m_distances = ma.masked_where(distances == 0, distances)
        min_distances = m_distances.min(0)
        min_distances[::-1].sort()
        self.plot_on_bokeh_hist('closest_neighbor.html', 'Trajectories -->', 'Custom Distance (~ degrees)', 'Closest Neighbor', min_distances, [])

    def plot_on_bokeh_hist(self, filename, x_label, y_label, title, hist, edges):
        from bokeh.layouts import gridplot
        from bokeh.plotting import figure, show, output_file

        f = figure(title=title, tools="save", background_fill_color="#FFFFFF")

        if len(edges) > 0 :
            f.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="#33b5e5", line_color="#4285F4")
        else:
            f.line(y=hist, x=np.arange(len(hist)), line_color="#4285F4")
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
        neighbors = radius_neighbors_graph(paths, radius=0.005)
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

    def custom_dist(self, paths):
        def min_dist(u,v):
            def min_diff(o):
                return min(abs(o[0] - o[2]), abs(o[1] - o[3]))
            u_dist = min_diff(u)
            v_dist = min_diff(v)
            if u_dist > 0 and v_dist>0:
                return min(min_diff(u), min_diff(v))
            if u_dist == 0 and v_dist > 0:
                return v_dist
            if v_dist == 0 and u_dist > 0:
                return u_dist
            if v_dist == 0 and v_dist ==0:
                return -1
            return min(min_diff(u), min_diff(v))
        def f(u, v):
            min_delta = min_dist(u,v)
            if min_delta > 0:
                return distance.euclidean(u,v)/min_dist(u,v)
            print('delta is zero')
            return 0
        distances = cdist(paths, paths, f)
        # distances = cdist(paths, paths, lambda u, v: np.sqrt(((u-v)**2).sum())/min_dist(u,v))
        return distances


if __name__ == '__main__':
    trajectory = Trajectory()
    # trajectory.distance_plot()
    # trajectory.neighbors_plot()
    trajectory.trajectories_dbscan("default")
    # trajectory.trajectories_hdbscan(2)
    print('done')
