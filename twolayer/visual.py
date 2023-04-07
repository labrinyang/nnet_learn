'''尝试画了一个可爱的神经网络图'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class MultiLayerNetVisualization:
    def __init__(self, multi_layer_net):
        self.multi_layer_net = multi_layer_net

    def visualize(self):
        layers = self.multi_layer_net.layers
        fig, ax = plt.subplots()

        max_neurons = max([layer['neurons'] for layer in layers])
        layer_width = 1 / (len(layers) + 1)

        for i, layer in enumerate(layers):
            x_offset = (i + 1) * layer_width
            y_space = 1 / (layer['neurons'] + 1)

            for neuron in range(layer['neurons']):
                y_offset = (neuron + 1) * y_space
                circle = mpatches.Circle((x_offset, y_offset), 0.03, color='blue', zorder=4)
                ax.add_patch(circle)

                if i > 0:
                    prev_layer = layers[i - 1]
                    prev_y_space = 1 / (prev_layer['neurons'] + 1)

                    for prev_neuron in range(prev_layer['neurons']):
                        prev_y_offset = (prev_neuron + 1) * prev_y_space
                        line = plt.Line2D(
                            [x_offset - layer_width, x_offset],
                            [prev_y_offset, y_offset],
                            color='gray',
                            zorder=1
                        )
                        ax.add_line(line)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.show()


# 示例 MultiLayerNet 类
class MultiLayerNetExample:
    def __init__(self):
        self.layers = [
            {'neurons': 4, 'activation': 'input'},
            {'neurons': 5, 'activation': 'relu'},
            {'neurons': 3, 'activation': 'relu'},
            {'neurons': 1, 'activation': 'sigmoid'}
        ]


if __name__ == "__main__":
    example_net = MultiLayerNetExample()
    visualizer = MultiLayerNetVisualization(example_net)
    visualizer.visualize()
