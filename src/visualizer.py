import pickle
from collections import defaultdict
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import visdom
from plotly.subplots import make_subplots


class VisdomPlotter:
    def __init__(self, env_name="main", port=8097):
        self.vis = visdom.Visdom(env=env_name, port=port)
        self.env = env_name
        self.n_plots = 0

        # Hyperparams
        self.hyperparams = defaultdict(dict)
        self.hyperparam_win = None

        # Gradient flow
        self.gradient_flow_win = None
        self.gradient_by_epochs = pd.DataFrame(columns=["epoch", "gradients"])

        # Classic plots
        self.line_plots = dict()
        self.images = dict()
        self.texts = dict()
        self.tables = dict()

    def save_hyperparameters(self, hyperparams):
        if self.hyperparam_win is None:
            self.hyperparam_win = self.n_plots
            self.n_plots += 1

        for name, value in hyperparams.items():
            table_name, hp_name = name.split("/")
            self.hyperparams[table_name][hp_name] = value

    def add_data(self, y, plot_name: str, line_name: str):
        if plot_name not in self.line_plots:
            # metric_name -> (window_id, lines data)
            self.line_plots[plot_name] = (self.n_plots, defaultdict(list))
            self.n_plots += 1

        if (
            self.line_plots[plot_name][1][line_name] != []
            and np.array(y).shape != self.line_plots[plot_name][1][line_name][-1].shape
        ):
            self.line_plots[plot_name][1][line_name] = []
            print(f"Warning: {plot_name}/{line_name} has different shape, reset!")
        self.line_plots[plot_name][1][line_name].append(y)

    def add_images(self, images, name):
        if name not in self.images:
            self.images[name] = [self.n_plots, None]
            self.n_plots += 1

        self.images[name][1] = images

    def add_table(self, table: dict, name: str):
        if name not in self.tables:
            self.tables[name] = [self.n_plots, None]
            self.n_plots += 1

        self.tables[name][1] = table

    def add_gradient_flow(self, model: torch.nn.Module):
        self.layer_names = []
        gradient_by_layers = []

        if self.gradient_flow_win is None:
            self.gradient_flow_win = self.n_plots
            self.n_plots += 1

        for name, params in model.named_parameters():
            if params.requires_grad and ("bias" not in name):
                self.layer_names.append(name)
                params = params.cpu().detach().flatten().numpy()
                distrib = np.random.choice(params, size=200)
                gradient_by_layers.append(distrib)

        gradient_by_layers = np.array(gradient_by_layers)  # [n_layers, n_bins]
        gradient_by_epochs = np.mean(gradient_by_layers, axis=0)  # [n_bins,]
        epoch_id = (
            0
            if len(self.gradient_by_epochs) == 0
            else self.gradient_by_epochs["epoch"].max() + 1
        )
        df_epoch = pd.DataFrame(
            {
                "epoch": epoch_id,
                "gradients": gradient_by_epochs,
            }
        )
        df_epoch = pd.concat((self.gradient_by_epochs, df_epoch))
        if df_epoch["epoch"].nunique() > 5:
            df_epoch = df_epoch[df_epoch["epoch"] != df_epoch["epoch"].min()]

        self.gradient_by_epochs = df_epoch

    def add_text(self, text: str, name: str):
        if name not in self.texts:
            self.texts[name] = [self.n_plots, None]
            self.n_plots += 1

        self.texts[name][1] = text

    def update_line_plots(self):
        for metric_name, (win_id, lines) in self.line_plots.items():
            Y, names = [], []
            for line_name, data in lines.items():
                Y.append(data)
                names.append(line_name)

            try:
                Y = np.array(Y).T  # [n_points, n_lines]
            except ValueError:
                traceback.print_exc()
                print("Skip graph %s" % str(names))
                continue

            X = np.arange(len(Y))
            self.vis.line(
                Y,
                X=X,
                win=win_id,
                opts={"legend": names, "title": metric_name},
            )

    def update_image_plots(self):
        for name, (win_id, images) in self.images.items():
            self.vis.images(
                images,
                win=win_id,
                env=self.env,
                opts={"title": name},
                nrow=max(len(images) // 2, 1),
                padding=10,
            )

    def update_texts(self):
        for _, (win_id, text) in self.texts.items():
            self.vis.text(text, win=win_id)

    def update_tables(self):
        for name, (win_id, table) in self.tables.items():
            keys, values = [], []
            for key, value in table.items():
                keys.append(key)
                values.append(value)

            table = go.Table(
                header={"values": ["Name", "Value"]},
                cells={"values": [keys, values]},
            )

            table = go.Figure(data=[table])
            table.update_layout(
                height=100 + 120 * len(keys),
                width=400,
                title_text=name,
            )

            self.vis.plotlyplot(table, win=win_id)

    def update_hyperparam_plots(self):
        if self.hyperparam_win is None:
            return

        fig = make_subplots(
            rows=len(self.hyperparams),
            cols=1,
            specs=[[{"type": "table"}] for _ in range(len(self.hyperparams))],
            subplot_titles=[n for n in self.hyperparams.keys()],
        )

        for i, table_data in enumerate(self.hyperparams.values()):
            keys, values = [], []
            for key, value in table_data.items():
                keys.append(key)
                values.append(value)

            table = go.Table(
                header={"values": keys},
                cells={"values": [[v] for v in values]},
            )

            fig.add_trace(table, row=i + 1, col=1)

        fig.update_layout(
            height=100 + 100 * len(self.hyperparams),
            width=1000,
            title_text="Hyperparameters",
        )
        self.vis.plotlyplot(fig, win=self.hyperparam_win)

    def update_gradient_flow_plots(self):
        if self.gradient_flow_win is None:
            return

        fig = px.violin(self.gradient_by_epochs, x="epoch", y="gradients")
        self.vis.plotlyplot(fig, win=self.gradient_flow_win)

    def update(self):
        self.update_line_plots()
        self.update_image_plots()
        self.update_hyperparam_plots()
        self.update_gradient_flow_plots()
        self.update_texts()
        self.update_tables()

    def save(self, dir_name: Path):
        with open(dir_name / "visdom.pkl", "wb") as pickle_file:
            data = {
                "env_name": self.env,
                "n_plots": self.n_plots,
                "line_plots": self.line_plots,
                "images": self.images,
                "texts": self.texts,
                "hyperparam_win": self.hyperparam_win,
                "hyperparams": self.hyperparams,
            }
            pickle.dump(data, pickle_file)

    @staticmethod
    def load(pickle_path, env_name: str):
        vis = None
        vis = VisdomPlotter(env_name)
        with open(pickle_path, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            vis.n_plots = data["n_plots"]
            vis.line_plots = data["line_plots"]
            vis.images = data["images"]
            vis.hyperparam_win = data["hyperparam_win"]
            vis.hyperparams = data["hyperparams"]
            vis.texts = data["texts"]
        return vis
