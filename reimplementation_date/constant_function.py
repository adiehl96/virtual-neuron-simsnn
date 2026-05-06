from simsnn.core.networks import Network
from simsnn.core.simulators import Simulator
import numpy as np
from snnsimsnn.concepts.reimplementation_date.virtual_neuron import VirtualNeuron
from snnsimsnn.func import (
    date_to_dec,
    raster_to_bin_list_neg,
    raster_to_bin_list_pos,
)


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

    # Constant
    constant_pos = np.random.randint(2, size=positive_precision)
    constant_neg = np.random.randint(2, size=negative_precision)

    print("\nPositive Constant:", constant_pos)
    print("Negative Constant:", constant_neg)

    # Random numbers
    number = np.random.randint(2, size=(positive_precision + negative_precision))

    # A and B for this iteration
    a_pos = constant_pos
    a_neg = constant_neg
    b_pos = number[0:positive_precision]
    b_neg = number[positive_precision:]

    print("a_pos:", a_pos)
    print("b_pos:", b_pos)
    print("a_neg:", a_neg)
    print("b_neg:", b_neg)

    # Create virtual neurons
    A = VirtualNeuron(net, precision)
    B = VirtualNeuron(net, precision)
    C = VirtualNeuron(net, precision)

    connect_virtual_neurons_constant(A, B, C)

    spike_generator = net.createInputTrain([1], loop=False, ID="spike_generator")
    for i in range(positive_precision):
        if a_pos[positive_precision - i - 1] == 1:
            net.createSynapse(
                pre=spike_generator,
                post=A.x_positive[i],
                ID=f"{spike_generator.ID}_{A.x_positive[i].ID}",
                w=1,
                d=1,
            )

        if b_pos[positive_precision - i - 1] == 1:
            net.createSynapse(
                pre=spike_generator,
                post=B.x_positive[i],
                ID=f"{spike_generator.ID}_{B.x_positive[i].ID}",
                w=1,
                d=1,
            )

    for i in range(negative_precision):
        if a_neg[negative_precision - i - 1] == 1:
            net.createSynapse(
                pre=spike_generator,
                post=A.x_negative[i],
                ID=f"{spike_generator.ID}_{A.x_negative[i].ID}",
                w=1,
                d=1,
            )

        if b_neg[negative_precision - i - 1] == 1:
            net.createSynapse(
                pre=spike_generator,
                post=B.x_negative[i],
                ID=f"{spike_generator.ID}_{B.x_negative[i].ID}",
                w=1,
                d=1,
            )

    sim.raster.addTarget(list(C.z_positive.values()))
    sim.raster.addTarget(list(C.z_negative.values()))

    duration = (positive_precision + negative_precision) * 2
    sim.run(duration, plotting=plotting)

    # Display positive spike recorders
    print("\nPositive Results:")

    read_out_time = 22
    lbr = positive_precision + 1
    rasterdata = sim.raster.get_measurements()

    if positive_precision > 0:
        a_pos_dec = date_to_dec(a_pos, precision[0], precision[1])
        b_pos_dec = date_to_dec(b_pos, precision[0], precision[1])
        c_pos = raster_to_bin_list_pos(lbr, rasterdata, read_out_time)
        c_pos_dec = date_to_dec(c_pos, precision[0], precision[1])

        print("A:", a_pos_dec)
        print("B:", b_pos_dec)
        print("A + B =", a_pos_dec + b_pos_dec)
        print("C:", c_pos_dec)

    print("\nNegative Results:")

    if negative_precision > 0:
        print("a_neg", a_neg)
        a_neg_dec = date_to_dec(a_neg, precision[2], precision[3], neg=True)
        b_neg_dec = date_to_dec(b_neg, precision[2], precision[3], neg=True)
        c_neg = raster_to_bin_list_neg(lbr, rasterdata, read_out_time)
        c_neg_dec = date_to_dec(c_neg, precision[2], precision[3], neg=True)

        print("A:", a_neg_dec)
        print("B:", b_neg_dec)
        print("C:", c_neg_dec)

    return (c_pos_dec == a_pos_dec) and (c_neg_dec == a_neg_dec)


def connect_virtual_neurons_constant(A, B, C):
    """Connects two input neurons A and B to output neuron C
    Params:
        A: VirtualNeuron, first input neuron
        B: VirtualNeuron, second input neuron
        C: VirtualNeuron, output neuron
    """

    # Check precision compatibility of A and B
    if A.positive_precision > C.positive_precision:
        raise ValueError(
            "Positive precision of output virtual neuron is less than first input virtual neuron"
        )

    if A.positive_precision > B.positive_precision:
        raise ValueError(
            "Positive precision of output virtual neuron is less than second input virtual neuron"
        )

    if A.negative_precision > C.negative_precision:
        raise ValueError(
            "Negative precision of output virtual neuron is less than first input virtual neuron"
        )

    if A.negative_precision > B.negative_precision:
        raise ValueError(
            "Negative precision of output virtual neuron is less than second input virtual neuron"
        )

    if not (A.net is B.net and B.net is C.net and A.net is C.net):
        raise ValueError(
            "Virtual Neurons to be connected need to be in the same network."
        )

    net = A.net

    # Connect A to C
    for i in range(A.positive_precision):
        net.createSynapse(
            pre=A.z_positive[i],
            post=C.x_positive[i],
            ID=f"{A.z_positive[i].ID}_{C.x_positive[i].ID}",
            w=1,
            d=1,
        )

    for i in range(A.negative_precision):
        net.createSynapse(
            pre=A.z_negative[i],
            post=C.x_negative[i],
            ID=f"{A.z_negative[i].ID}_{C.x_negative[i].ID}",
            w=1,
            d=1,
        )

    # Connect B to C
    for i in range(B.positive_precision):
        net.createSynapse(
            pre=B.z_positive[i],
            post=C.y_positive[i],
            ID=f"{A.z_positive[i].ID}_{C.y_positive[i].ID}",
            w=0,
            d=1,
        )

    for i in range(B.negative_precision):
        net.createSynapse(
            pre=B.z_negative[i],
            post=C.y_negative[i],
            ID=f"{A.z_negative[i].ID}_{C.y_negative[i].ID}",
            w=0,
            d=1,
        )
