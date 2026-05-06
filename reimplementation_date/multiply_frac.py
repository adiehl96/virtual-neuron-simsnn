from os import system
from simsnn.core.networks import Network
from simsnn.core.simulators import Simulator
import numpy as np
from snnsimsnn.concepts.reimplementation_date.n_addition import connect_n_neurons
from snnsimsnn.concepts.reimplementation_date.virtual_neuron import VirtualNeuron
from snnsimsnn.func import (
    date_to_dec,
    raster_to_bin_list_neg,
    raster_to_bin_list_pos,
    bin_string_to_bin_list,
    bin_list_to_nat,
)
import sys
from pprint import pprint


def run(x, y, plotting=False):
    net = Network()
    sim = Simulator(net)

    # Precision (we can only have fractional precision,
    # as integer precision doesn't work in this calculation)
    precision = [0, 8, 0, 8]  # 16-bit fractional precision
    positive_precision = precision[0] + precision[1]
    negative_precision = precision[2] + precision[3]

    print("Total Precision:", precision)
    print("Positive Precision:", positive_precision, "bits")
    print("Negative Precision:", negative_precision, "bits")

    # A and B for this iteration
    a_pos = encode(x, positive_precision)
    print(a_pos)
    print(bin_list_to_nat(a_pos))
    a_neg = encode(0, negative_precision)
    b_pos = encode(y, positive_precision)
    b_neg = encode(0, negative_precision)

    # Create virtual neurons
    X = VirtualNeuron(net, precision)
    Y = VirtualNeuron(net, precision)

    # sys.exit(0)

    create_and_connect_spike_gen(
        [a_pos, b_pos],
        [a_neg, b_neg],
        [X, Y],
        positive_precision,
        negative_precision,
        net,
    )

    # pass_vn_to_vns(A, C)
    # pass_vn_to_vn(A, C)
    BG = pass_and_shift_vn_to_bg(X)
    demux_vn_to_bg(Y, BG)
    IG, read_out = pass_bg_to_ig(net, precision, BG)

    # sim.raster.addTarget(list(X.z_positive.values()))
    # sim.raster.addTarget(list(Y.z_positive.values()))
    sim.raster.addTarget(list(read_out.z_positive.values()))
    # sim.raster.addTarget(BG[1][2])
    # sim.multimeter.addTarget(BG[1][2])
    # sim.multimeter.addTarget()
    # sim.raster.addTarget(list(IG[1].z_positive.values()))
    # sim.raster.addTarget(list(read_out.z_negative.values()))

    # duration = (positive_precision + negative_precision) * 2
    sim.run(105, plotting=plotting)
    rasterdata = sim.raster.get_measurements()
    lbr = positive_precision
    read_out_time = 100
    c_pos = raster_to_bin_list_pos(lbr, rasterdata, read_out_time)
    c_pos_dec = date_to_dec(c_pos, precision[0], precision[1])
    print(c_pos_dec)
    if x * y == c_pos_dec:
        print(f"Success: {x} * {y} = {c_pos_dec}")
    else:
        print(f"Error: {x} * {y} = {c_pos_dec}")


def create_bg(net, n, precision):
    """
    Create n block groups consisting of 
    """
    bg = [
        [net.createLIF(ID=f"bg_neuron_{j}_{i}", thr=2) for i in range(precision[0])]
        for j in range(n)
    ]
    return bg


def create_ig(net, n, precision):
    ig = [VirtualNeuron(net, precision) for _ in range(n)]
    return ig


def pass_vn_to_bg(X, BG=None):
    """
    Pass n-precision Virtual Neuron value to n Block Groups
    X: Virtual Neuron
    BG: List of Lists containing n LIF neurons
    """
    net = X.net
    BG = create_bg(net, X.precision[0], X.precision) if BG is None else BG
    for i in range(X.precision[0]):
        for j in range(X.precision[0]):
            net.createSynapse(
                pre=X.z_positive[j],
                post=BG[i][j],
                ID=f"{X.z_positive[j].ID}_{BG[i][j].ID}",
                w=1,
                d=1,
            )
    return BG


def pass_and_shift_vn_to_bg(X, BG=None):
    """
    Pass n-precision Virtual Neuron value to n Block Groups,
    while shifting the connections up with increasing index of the block group
    X: Virtual Neuron
    BG: List of n Lists containing n LIF neurons
    """

    net = X.net
    BG = create_bg(net, X.precision[0], X.precision) if BG is None else BG
    for i in range(X.precision[0]):
        for j in range(X.precision[0]):
            if i + j < X.positive_precision:
                net.createSynapse(
                    pre=X.z_positive[j],
                    post=BG[i][j + i],
                    ID=f"{X.z_positive[j].ID}_{BG[i][j].ID}",
                    w=1,
                    d=1,
                )
    return BG


def demux_vn_to_bg(X, BG=None):
    """
    Connect outgoing neurons of n-precision Virtual Neuron one-to-many to Block groups
    X: Virtual Neuron
    BG: List of n Lists containing n LIF neurons
    """

    net = X.net
    BG = create_bg(X.precision[0], X.precision) if BG is None else BG
    for i in range(X.precision[0]):
        for j in range(X.precision[0]):
            net.createSynapse(
                pre=X.z_positive[i],
                post=BG[i][j],
                ID=f"{X.z_positive[j].ID}_{BG[i][j].ID}",
                w=1,
                d=1,
            )
    return BG


def pass_bg_to_ig(net, precision, BG, IG=None):
    """
    net: the spiking neural network containing the lif neurons
    precision: List of 4 ints denoting positive integer precision,
        positive fractional precision, negative integer precision
        and negative fractional precision
    BG: List of n Lists containing n LIF neurons
    IG: List of n Virtual Neurons
    """

    ig_size = int(np.ceil(int("1" * precision[0], 2) / 2))
    IG = create_ig(net, ig_size, precision) if IG is None else IG
    for i in range(len(BG)):
        for j in range(precision[0]):
            if (
                not j % 2
            ):  # alternatingly connect to x or y component of the virtual neuron
                net.createSynapse(
                    pre=BG[i][j],
                    post=IG[i].x_positive[j],
                    ID=f"{BG[i][j].ID}_{IG[i].x_positive[j].ID}",
                    w=1,
                    d=1,
                )
            else:
                net.createSynapse(
                    pre=BG[i][j],
                    post=IG[i].y_positive[j],
                    ID=f"{BG[i][j].ID}_{IG[i].y_positive[j].ID}",
                    w=1,
                    d=1,
                )
    read_out = connect_n_neurons(IG)
    return IG, read_out


def pass_and_shift_vn_to_vns(A, B):
    """Connects input neuron A to list of virtual neurons B
    Params:
        A: VirtualNeuron, first input neuron
        B: VirtualNeuron, list of neurons
    """

    net = A.net

    # Connect A to B
    for idx, b in enumerate(B):
        for i in range(A.positive_precision):
            if i + idx < A.positive_precision:
                net.createSynapse(
                    pre=A.z_positive[i],
                    post=b.x_positive[i + idx],
                    ID=f"{A.z_positive[i].ID}_{b.x_positive[i].ID}",
                    w=1,
                    d=1,
                )

        for i in range(A.negative_precision):
            if i + idx < A.negative_precision:
                net.createSynapse(
                    pre=A.z_negative[i],
                    post=b.x_negative[i + idx],
                    ID=f"{A.z_negative[i].ID}_{b.x_negative[i].ID}",
                    w=1,
                    d=1,
                )


def create_and_connect_spike_gen(
    pos_list, neg_list, neuron_list, positive_precision, negative_precision, net
):
    spike_generator = net.createInputTrain([1], loop=False, ID="spike_generator")
    for pos_val, neuron in zip(pos_list, neuron_list):
        for i in range(positive_precision):
            if pos_val[positive_precision - i - 1] == 1:
                net.createSynapse(
                    pre=spike_generator,
                    post=neuron.x_positive[i],
                    ID=f"{spike_generator.ID}_{neuron.x_positive[i].ID}",
                    w=1,
                    d=1,
                )

    for neg_val, neuron in zip(neg_list, neuron_list):
        for i in range(negative_precision):
            if neg_val[negative_precision - i - 1] == 1:
                net.createSynapse(
                    pre=spike_generator,
                    post=neuron.x_negative[i],
                    ID=f"{spike_generator.ID}_{neuron.x_negative[i].ID}",
                    w=1,
                    d=1,
                )
