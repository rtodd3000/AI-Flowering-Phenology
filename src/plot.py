from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator, FuncFormatter
from collections import defaultdict

COLORS = ["#d80000", "#000cf1", "#00de16", "#e68200", "#ce00b9", "#00b0cf", "#464546", "#00eda6"]

class FloweringPlot:
    def __init__(self):
        self.marker = None
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        weekly_labels = [f"{m} Week {w}" for m in self.months for w in range(1, 5)]

        def month_formatter(val, pos):
            idx = int(round(val))
            if idx % 4 == 0 and 0 <= idx < len(weekly_labels):
                return self.months[idx // 4]
            return ""

        self.graph = Figure(figsize=(5, 5), dpi=100)
        self.graph.suptitle("Flowering Intensity Over Time", fontsize=16)

        self.plot1 = self.graph.add_subplot(111)
        self.graph.subplots_adjust(left=0.08, right=0.88, top=0.9, bottom=0.15)
        self.plot1.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        self.plot1.set_xlabel("Date")
        self.plot1.set_ylabel("Intensity")

        self.plot1.set_yticks([0, 1, 2, 3])
        self.plot1.set_ylim(-0.2, 3.2)
        self.plot1.set_xlim(-1, 48)

        self.plot1.yaxis.set_minor_locator(MultipleLocator(0.5))
        self.plot1.xaxis.set_major_locator(MultipleLocator(4))
        self.plot1.xaxis.set_minor_locator(MultipleLocator(1))
        self.plot1.tick_params(which="major", length=8, color="black")
        self.plot1.tick_params(which="minor", length=4, color="gray")
        self.plot1.xaxis.set_major_formatter(FuncFormatter(month_formatter))

    def plot(self, raw_data):
    # Remove existing plot lines
        for line in self.plot1.get_lines():
            if line is not self.marker:
                line.remove()
            self.marker = None  # reset marker state since it was removed

        # Scrub and group
        grouped = defaultdict(list)
        for row in raw_data:
            date, intensity = row[1], int(row[2].split()[0])
            parts = date.split("-")
            year = parts[0]
            x_pos = (int(parts[1]) - 1) * 4 + int(parts[2]) // 7
            grouped[year].append([x_pos, intensity])

        # Plot each year
        for i, (year, entries) in enumerate(sorted(grouped.items())):
            entries.sort(key=lambda e: e[0])
            x_vals = [entry[0] for entry in entries]
            y_vals = [entry[1] for entry in entries]
            self.plot1.plot(
                x_vals, y_vals,
                label     = year,
                color     = COLORS[i % len(COLORS)],
                linewidth = 2,
                marker     = "o",
                markersize = 2
            )
        self.plot1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

    def update_point(self, date, new_intensity):
        parts = date.split("-")
        x_pos = (int(parts[1]) - 1) * 4 + int(parts[2]) // 7
        for line in self.plot1.get_lines():
            if line.get_label() == str(parts[0]):
                x_vals = list(line.get_xdata())
                y_vals = list(line.get_ydata())
                if x_pos in x_vals:
                    idx = x_vals.index(x_pos)
                    y_vals[idx] = int(new_intensity)
                    line.set_ydata(y_vals)
                break

    def set_marker(self, x_pos, intensity):
        if self.marker:
            self.marker.remove()
        self.marker, = self.plot1.plot(x_pos, int(intensity), marker="o", markersize=5, color="black")

    def save(self, file_path):
        if self.marker:
            self.marker.set_visible(False)
            self.graph.savefig(file_path)
            self.marker.set_visible(True)
        else:
            self.graph.savefig(file_path)

    def get_figure (self):
        return self.graph