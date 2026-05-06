import sys
from simsnn.core.networks import Network
from simsnn.core.simulators import Simulator
import numpy as np
from snnsimsnn.concepts.reimplementation_date.virtual_neuron import (
    VirtualNeuron,
    connect_virtual_neurons,
)
from snnsimsnn.func import (
    date_to_dec,
    raster_to_bin_list_neg,
    raster_to_bin_list_pos,
)
from pprint import pprint


def run(plotting=False):
    net = Network()
    sim = Simulator(net)

    # Precision
    precision = [8, 0, 8, 0]  # 16-bit integer precision
    positive_precision = precision[0] + precision[1]
    negative_precision = precision[2] + precision[3]

    print("Total Precision:", precision)
    print("Positive Precision:", positive_precision, "bits")
    print("Negative Precision:", negative_precision, "bits")

    # Random numbers
    a_pos = np.random.randint(2, size=positive_precision)
    a_pos[0] = 0
    a_neg = np.random.randint(2, size=negative_precision)
    a_neg[0] = 0
    b_pos = np.random.randint(2, size=positive_precision)
    b_pos[0] = 0
    b_neg = np.random.randint(2, size=negative_precision)
    b_neg[0] = 0
    d_pos = np.random.randint(2, size=positive_precision)
    d_pos[0] = 0
    d_neg = np.random.randint(2, size=negative_precision)
    d_neg[0] = 0
    e_pos = np.random.randint(2, size=positive_precision)
    e_pos[0] = 0
    e_neg = np.random.randint(2, size=negative_precision)
    e_neg[0] = 0

    print("a_pos:", a_pos)
    print("b_pos:", b_pos)
    print("a_neg:", a_neg)
    print("b_neg:", b_neg)

    # Create virtual neurons
    A = VirtualNeuron(net, precision)
    B = VirtualNeuron(net, precision)
    D = VirtualNeuron(net, precision)
    E = VirtualNeuron(net, precision)

    read_out = connect_n_neurons([A, B, D, E])

    create_and_connect_spike_gen(
        [a_pos, b_pos, d_pos, e_pos],
        [a_neg, b_neg, d_neg, e_neg],
        [A, B, D, E],
        positive_precision,
        negative_precision,
        net,
    )

    sim.raster.addTarget(list(read_out.z_positive.values()))
    sim.raster.addTarget(list(read_out.z_negative.values()))

    duration = (positive_precision + negative_precision) * 3
    sim.run(duration, plotting=plotting)

    # Display positive spike recorders
    print("\nPositive Results:")

    read_out_time = 33
    lbr = positive_precision + 1
    rasterdata = sim.raster.get_measurements()

    if positive_precision > 0:
        a_pos_dec = date_to_dec(a_pos, precision[0], precision[1])
        b_pos_dec = date_to_dec(b_pos, precision[0], precision[1])
        d_pos_dec = date_to_dec(d_pos, precision[0], precision[1])
        e_pos_dec = date_to_dec(e_pos, precision[0], precision[1])
        read_out_pos = raster_to_bin_list_pos(lbr, rasterdata, read_out_time)
        read_out_pos_dec = date_to_dec(read_out_pos, precision[0], precision[1])

        print("A:", a_pos_dec)
        print("B:", b_pos_dec)
        print("D:", d_pos_dec)
        print("E:", e_pos_dec)
        print("A + B + D + E =", a_pos_dec + b_pos_dec + d_pos_dec + e_pos_dec)
        print("read_out_pos_dec:", read_out_pos_dec)

    print("\nNegative Results:")

    if negative_precision > 0:
        print("a_neg", a_neg)
        a_neg_dec = date_to_dec(a_neg, precision[2], precision[3], neg=True)
        b_neg_dec = date_to_dec(b_neg, precision[2], precision[3], neg=True)
        d_neg_dec = date_to_dec(d_neg, precision[2], precision[3], neg=True)
        e_neg_dec = date_to_dec(e_neg, precision[2], precision[3], neg=True)

        read_out_neg = raster_to_bin_list_neg(lbr, rasterdata, read_out_time)
        read_out_neg_dec = date_to_dec(
            read_out_neg, precision[2], precision[3], neg=True
        )

        print("A:", a_neg_dec)
        print("B:", b_neg_dec)
        print("D:", d_neg_dec)
        print("E:", e_neg_dec)

        print("G:", read_out_neg_dec)

    return (read_out_pos_dec == a_pos_dec + b_pos_dec + d_pos_dec + e_pos_dec) and (
        read_out_neg_dec == a_neg_dec + b_neg_dec + d_neg_dec + e_neg_dec
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


def connect_n_neurons(neuron_list):
    net = neuron_list[0].net
    precision = neuron_list[0].precision
    neuronlists = [neuron_list]
    while len(neuronlists[-1]) > 1:
        neuronlists[-1] = [
            neuronlists[-1][i : i + 2] for i in range(0, len(neuronlists[-1]), 2)
        ]
        neuronlists.append([])
        for tuple in neuronlists[-2]:
            if len(tuple) == 2:
                neuronlists[-1].append(VirtualNeuron(net, precision))
                connect_virtual_neurons(tuple[0], tuple[1], neuronlists[-1][-1])
            if len(tuple) == 1:
                neuronlists[-1].append(tuple[0])
    return neuronlists[-1][0]
