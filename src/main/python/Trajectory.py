from pysal.cg.shapes import Point
from pysal.cg.shapes import Chain
import pysal

MAX_LINES = 10000
NUM_GROUPS = 100
API_KEY="AIzaSyDE74s0qo35vvq7jIs4zINqidd2z-6GqA0"

class Trajectory:
    def __init__(self):
        print("start")

    def get_tree(self, pts):
        tree = pysal.cg.kdtree.KDTree(pts, leafsize=10, distance_metric='Euclidean', radius=6371.0)
        return tree

    def plot_on_bokeh(self, starts, ends, bboxes_start, bboxes_end):
        from bokeh.io import output_file, show
        from bokeh.models import (
            GMapPlot, GMapOptions, ColumnDataSource, GeoJSONDataSource, Circle, Segment, Quad, Range1d, PanTool, WheelZoomTool, BoxSelectTool
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
                lon=[b.lower + b.height/2 for b in bboxes_start],
                lat=[b.left + b.width/2 for b in bboxes_start],
                radius=[max(b.width, b.height)/2 for b in bboxes_start]
            )
        )

        source_circles_end = ColumnDataSource(
            data=dict(
                lon=[b.lower + b.height/2 for b in bboxes_end],
                lat=[b.left + b.width/2 for b in bboxes_end],
                radius=[max(b.width, b.height)/2 for b in bboxes_end]
            )
        )

        circle = Circle(x="lon", y="lat", size=4, fill_color="blue", fill_alpha=0.8, line_color=None)
        plot.add_glyph(source_start, circle)

        circle = Circle(x="lon", y="lat", size=4, fill_color="green", fill_alpha=0.8, line_color=None)
        plot.add_glyph(source_end, circle)

        circle = Circle(x="lon", y="lat", size=20, fill_color="blue", fill_alpha=0.1, line_color=None)
        plot.add_glyph(source_circles_start, circle)

        circle = Circle(x="lon", y="lat", size=20, fill_color="green", fill_alpha=0.1, line_color=None)
        plot.add_glyph(source_circles_end, circle)

        plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
        output_file("/assets/gmap_plot.html")
        show(plot)

    def trajectories(self):
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

        bboxs_start = []
        bboxs_end = []

        for g in start_groups:
            c = Chain(g)
            bboxs_start.append(c.bounding_box)

        for g in end_groups:
            c = Chain(g)
            bboxs_end.append(c.bounding_box)

        self.plot_on_bokeh(start_pos, end_pos, bboxs_start, bboxs_end)

if __name__ == '__main__':
    trajectory = Trajectory()
    trajectory.trajectories()
    print('done')