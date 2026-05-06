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
    precision = [4, 4, 4, 4]  # 16-bit  precision
    res_prec = [5, 4, 5, 4]
    positive_precision = precision[0] + precision[1]
    negative_precision = precision[2] + precision[3]

    print("Total Precision:", precision)
    print("Positive Precision:", positive_precision, "bits")
    print("Negative Precision:", negative_precision, "bits")

    # Random numbers
    number = np.random.randint(2, size=(positive_precision + negative_precision))

    a_pos = number[0:positive_precision]
    a_neg = number[positive_precision:]

    print("a_pos:", a_pos)
    print("a_neg:", a_neg)

    # Create virtual neurons
    A = VirtualNeuron(net, precision)
    B = VirtualNeuron(net, precision)

    connect_virtual_neurons_multiply_neg_one(A, B)

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

    for i in range(negative_precision):
        if a_neg[negative_precision - i - 1] == 1:
            net.createSynapse(
                pre=spike_generator,
                post=A.x_negative[i],
                ID=f"{spike_generator.ID}_{A.x_negative[i].ID}",
                w=1,
                d=1,
            )
    sim.raster.addTarget(list(B.z_positive.values()))
    sim.raster.addTarget(list(B.z_negative.values()))

    duration = (positive_precision + negative_precision) * 2
    sim.run(duration, plotting=plotting)

    # Display positive spike recorders
    print("\nPositive Results:")

    read_out_time = 22
    lbr = positive_precision + 1
    rasterdata = sim.raster.get_measurements()

    if positive_precision > 0:
        a_pos_dec = date_to_dec(a_pos, precision[0], precision[1])
        b_pos = raster_to_bin_list_pos(lbr, rasterdata, read_out_time)
        b_pos_dec = date_to_dec(b_pos, precision[0], precision[1])

        print("A:", a_pos_dec)
        print("A * -1 =", a_pos_dec * (-1))
        print("B:", b_pos_dec)

        print("\nNegative Results:")

    if negative_precision > 0:
        a_neg_dec = date_to_dec(a_neg, precision[2], precision[3], neg=True)
        b_neg = raster_to_bin_list_neg(lbr, rasterdata, read_out_time)
        b_neg_dec = date_to_dec(b_neg, precision[2], precision[3], neg=True)

        print("A:", a_neg_dec)
        print("A * -1 =", a_neg_dec * (-1))
        print("B:", b_neg_dec)

    if (b_pos_dec == -1 * a_neg_dec) and (b_neg_dec == -1 * a_pos_dec):
        print("\nCorrect Results!")
    else:
        print("\nIncorrect Results!")

    return (b_pos_dec == -1 * a_neg_dec) and (b_neg_dec == -1 * a_pos_dec)


def connect_virtual_neurons_multiply_neg_one(A, B):
    """Connects input neuron A to read out neuron B. The negative multiplication is embedded in the connection
    Params:
        A: VirtualNeuron, first input neuron
        B: VirtualNeuron, second input neuron
        C: VirtualNeuron, output neuron
    """

    # Check precision compatibility of A and B

    if A.positive_precision > B.positive_precision:
        raise ValueError(
            "Positive precision of output virtual neuron is less than second input virtual neuron"
        )

    if A.negative_precision > B.negative_precision:
        raise ValueError(
            "Negative precision of output virtual neuron is less than second input virtual neuron"
        )

    if not (A.net is B.net):
        raise ValueError(
            "Virtual Neurons to be connected need to be in the same network."
        )

    net = A.net

    # Connect A to B
    for i in range(A.positive_precision):
        net.createSynapse(
            pre=A.z_positive[i],
            post=B.x_negative[i],
            ID=f"{A.z_positive[i].ID}_{B.x_positive[i].ID}",
            w=1,
            d=1,
        )

    for i in range(A.negative_precision):
        net.createSynapse(
            pre=A.z_negative[i],
            post=B.x_positive[i],
            ID=f"{A.z_negative[i].ID}_{B.x_positive[i].ID}",
            w=1,
            d=1,
        )
