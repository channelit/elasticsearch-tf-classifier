from pysal.cg.shapes import Point
from pysal.cg.shapes import Chain
import pysal

MAX_LINES = 1000
NUM_GROUPS = 100
API_KEY="AIzaSyDE74s0qo35vvq7jIs4zINqidd2z-6GqA0"

class Trajectory:
    def __init__(self):
        print("start")

    def get_tree(self, pts):
        tree = pysal.cg.kdtree.KDTree(pts, leafsize=10, distance_metric='Euclidean', radius=6371.0)
        return tree

    def plot_on_bokeh(self, lats_start, lons_start, lats_end, lons_end, bboxes_start, bboxes_end, circles_start, circles_end):
        from bokeh.io import output_file, show
        from bokeh.models import (
            GMapPlot, GMapOptions, ColumnDataSource, GeoJSONDataSource, Circle, Segment, Quad, Range1d, PanTool, WheelZoomTool, BoxSelectTool
        )
        from bokeh.resources import INLINE
        from bokeh.plotting import figure
        import bokeh.io
        bokeh.io.output_notebook(INLINE)

        map_options = GMapOptions(lat=40.73, lng=-73.56, map_type="roadmap", zoom=11)

        plot = GMapPlot(x_range=Range1d(), y_range=Range1d(), map_options=map_options)
        plot.title.text = "New York"
        plot.api_key = API_KEY

        source_start = ColumnDataSource(
            data=dict(
                lat=lats_start,
                lon=lons_start,
            )
        )
        source_end = ColumnDataSource(
            data=dict(
                lat=lats_end,
                lon=lons_end,
            )
        )

        source_bbox_start = ColumnDataSource(
            data=dict(
                top=bboxes_start[3],
                bottom=bboxes_start[2],
                left=bboxes_start[1],
                right=bboxes_start[0]
            )
        )

        source_bbox_end = ColumnDataSource(
            data=dict(
                top=bboxes_end[0],
                bottom=bboxes_end[1],
                left=bboxes_end[2],
                right=bboxes_end[3]
            )
        )

        source_circles_start = ColumnDataSource(
            data=dict(
                lat=[x[1] for x in (y[0] for y in circles_start)],
                lon=[x[0] for x in (y[0] for y in circles_start)],
                radius=[y[1] for y in circles_start]
            )
        )

        source_circles_end = ColumnDataSource(
            data=dict(
                lat=[x[1] for x in (y[0] for y in circles_end)],
                lon=[x[0] for x in (y[0] for y in circles_end)],
                radius=[y[1] for y in circles_end]
            )
        )

        circle = Circle(x="lon", y="lat", size=4, fill_color="blue", fill_alpha=0.8, line_color=None)
        plot.add_glyph(source_start, circle)

        circle = Circle(x="lon", y="lat", size=4, fill_color="green", fill_alpha=0.8, line_color=None)
        plot.add_glyph(source_end, circle)

        circle = Circle(x="lon", y="lat", size=20, fill_color="blue", fill_alpha=0.2, line_color=None)
        plot.add_glyph(source_circles_start, circle)

        circle = Circle(x="lon", y="lat", size=20, fill_color="green", fill_alpha=0.2, line_color=None)
        plot.add_glyph(source_circles_end, circle)


        # quad = Quad(top="top", bottom="bottom", left="left", right="right", fill_color="blue", fill_alpha=1)
        # segment = Segment(x1="top", x0="bottom", y1="left", y0="right", line_color="blue", line_alpha=1)

        # p.quad(top="top", bottom="bottom", left="left", right="right", color="blue", alpha=1, source=source_bbox_start)
        # f.segment(x1="left", x0="right", y1="top", y0="bottom", color="blue", source=source_bbox_start)
        # output_file("/assets/geojson.html")
        # show(f)
        # plot.add_glyph(source_bbox_start, segment)

        # quad = Quad(top="top", bottom="bottom", left="left", right="right", fill_color="blue", fill_alpha=0.9)
        # plot.add_glyph(source_bbox_end, quad)

        plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
        output_file("/assets/gmap_plot.html")
        show(plot)

    def trajectories(self):
        file = "/assets/yellow_tripdata_2015-12.csv"
        import csv

        start_pos = []
        end_pos = []

        lats_start = []
        lats_end = []
        lons_start = []
        lons_end = []

        with open(file) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')

            linectr = 0
            for row in readCSV:
                if 0 < linectr < MAX_LINES:

                    lats_start.append(float(row[6]))
                    lats_end.append(float(row[10]))
                    lons_start.append(float(row[5]))
                    lons_end.append(float(row[9]))

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


        lefts_start = []
        lefts_end = []

        rights_start = []
        rights_end = []

        uppers_start = []
        uppers_end = []

        lowers_start = []
        lowers_end = []

        circles_start = []
        circles_end = []

        for b in bboxs_start:
            lefts_start.append(b.left)
            rights_start.append(b.right)
            lowers_start.append(b.lower)
            uppers_start.append(b.upper)
            circles_start.append([[b.lower + b.height/2, b.left + b.width/2], max(b.width, b.height)])
        for b in bboxs_end:
            lefts_end.append(b.left)
            rights_end.append(b.right)
            lowers_end.append(b.lower)
            uppers_end.append(b.upper)
            circles_end.append([[b.lower + b.height/2, b.left + b.width/2], max(b.width, b.height)])

        bboxes_start=[uppers_start, lowers_start, lefts_start, rights_start]

        bboxes_end=[uppers_end, lowers_end, lefts_end, rights_end]

        self.plot_on_bokeh(lats_start, lons_start, lats_end, lons_end, bboxes_start, bboxes_end, circles_start, circles_end)

if __name__ == '__main__':
    trajectory = Trajectory()
    trajectory.trajectories()
    print('done')