class VirtualNeuron:
    """Virtual neuron mimicks the behavior of an artificial neuron using a collection of spiking neurons."""

    def __init__(self, net, precision=[4, 4, 4, 4]):
        """
        Initializes a virtual neuron object
        Params:
                precision: List of 4 ints denoting positive integer precision, positive fractional precision, negative integer precision and negative fractional precision
        """
        self.precision = precision
        self.positive_precision = precision[0] + precision[1]
        self.negative_precision = precision[2] + precision[3]
        self.total_precision = sum(precision)
        self.higher_precision = max(self.positive_precision, self.negative_precision)
        self.net = net

        # Setup incoming neurons
        self.x_positive = {}
        self.x_negative = {}
        self.y_positive = {}
        self.y_negative = {}

        for i in range(self.positive_precision):
            self.x_positive[i] = net.createLIF(ID=f"x_positive_{i}")
            self.y_positive[i] = net.createLIF(ID=f"y_positive_{i}", thr=1)

        for i in range(self.negative_precision):
            self.x_negative[i] = net.createLIF(ID=f"x_negative_{i}")
            self.y_negative[i] = net.createLIF(ID=f"y_negative_{i}")

        # Setup positive bit neurons
        self.bits_positive = {}
        if self.positive_precision > 0:
            for i in range(self.positive_precision + 1):
                self.bits_positive[i] = {}
                self.bits_positive[i][0] = net.createLIF(
                    ID=f"bits_positive_{i}_0", thr=1
                )
                self.bits_positive[i][1] = net.createLIF(
                    ID=f"bits_positive_{i}_1", thr=2
                )

                if i > 0:
                    self.bits_positive[i][2] = net.createLIF(
                        ID="bits_positive_{i}_2", thr=3
                    )

        # Setup negative bit neurons
        self.bits_negative = {}
        if self.negative_precision > 0:
            for i in range(self.negative_precision + 1):
                self.bits_negative[i] = {}
                self.bits_negative[i][0] = net.createLIF(
                    ID=f"bits_negative_{i}_0", thr=1
                )
                self.bits_negative[i][1] = net.createLIF(
                    ID=f"bits_negative_{i}_1", thr=2
                )

                if i > 0:
                    self.bits_negative[i][2] = net.createLIF(
                        ID=f"bits_negative_{i}_2", thr=3
                    )

        # Setup outgoing neurons
        self.z_positive = {}
        self.z_negative = {}

        if self.positive_precision > 0:
            for i in range(self.positive_precision + 1):
                self.z_positive[i] = net.createLIF(ID=f"z_positive_{i}", thr=1)

        if self.negative_precision > 0:
            for i in range(self.negative_precision + 1):
                self.z_negative[i] = net.createLIF(ID=f"z_negative_{i}", thr=1)

        # Neurons created
        # print("Neurons created...")

        # Setup synapses between positive incoming neurons and positive bit neurons
        for i in range(self.positive_precision):
            net.createSynapse(
                pre=self.x_positive[i],
                post=self.bits_positive[i][0],
                ID=f"{self.x_positive[i].ID}_{self.bits_positive[i][0].ID}",
                w=1,
                d=i + 1,
            )
            net.createSynapse(
                pre=self.x_positive[i],
                post=self.bits_positive[i][1],
                ID=f"{self.x_positive[i].ID}_{self.bits_positive[i][1].ID}",
                w=1,
                d=i + 1,
            )

            net.createSynapse(
                pre=self.y_positive[i],
                post=self.bits_positive[i][0],
                ID=f"{self.y_positive[i].ID}_{self.bits_positive[i][0].ID}",
                w=1,
                d=i + 1,
            )
            net.createSynapse(
                pre=self.y_positive[i],
                post=self.bits_positive[i][1],
                ID=f"{self.y_positive[i].ID}_{self.bits_positive[i][1].ID}",
                w=1,
                d=i + 1,
            )

            if i > 0:
                net.createSynapse(
                    pre=self.x_positive[i],
                    post=self.bits_positive[i][2],
                    ID=f"{self.x_positive[i].ID}_{self.bits_positive[i][2].ID}",
                    w=1,
                    d=i + 1,
                )
                net.createSynapse(
                    pre=self.y_positive[i],
                    post=self.bits_positive[i][2],
                    ID=f"{self.y_positive[i].ID}_{self.bits_positive[i][2].ID}",
                    w=1,
                    d=i + 1,
                )

        # Setup synapses between negative incoming neurons and negative bit neurons
        for i in range(self.negative_precision):
            net.createSynapse(
                pre=self.x_negative[i],
                post=self.bits_negative[i][0],
                ID=f"{self.x_negative[i].ID}_{self.bits_negative[i][0].ID}",
                w=1,
                d=i + 1,
            )
            net.createSynapse(
                pre=self.x_negative[i],
                post=self.bits_negative[i][1],
                ID=f"{self.x_negative[i].ID}_{self.bits_negative[i][1].ID}",
                w=1,
                d=i + 1,
            )

            net.createSynapse(
                pre=self.y_negative[i],
                post=self.bits_negative[i][0],
                ID=f"{self.y_negative[i].ID}_{self.bits_negative[i][0].ID}",
                w=1,
                d=i + 1,
            )
            net.createSynapse(
                pre=self.y_negative[i],
                post=self.bits_negative[i][1],
                ID=f"{self.y_negative[i].ID}_{self.bits_negative[i][1].ID}",
                w=1,
                d=i + 1,
            )

            if i > 0:
                net.createSynapse(
                    pre=self.x_negative[i],
                    post=self.bits_negative[i][2],
                    ID=f"{self.x_negative[i].ID}_{self.bits_negative[i][2].ID}",
                    w=1,
                    d=i + 1,
                )
                net.createSynapse(
                    pre=self.y_negative[i],
                    post=self.bits_negative[i][2],
                    ID=f"{self.y_negative[i].ID}_{self.bits_negative[i][2].ID}",
                    w=1,
                    d=i + 1,
                )

        # Setup carry synapses in positive bits
        for i in range(self.positive_precision):
            net.createSynapse(
                pre=self.bits_positive[i][1],
                post=self.bits_positive[i + 1][0],
                ID=f"{self.bits_positive[i][1].ID}_{self.bits_positive[i+1][0].ID}",
                w=1,
                d=1,
            )
            net.createSynapse(
                pre=self.bits_positive[i][1],
                post=self.bits_positive[i + 1][1],
                ID=f"{self.bits_positive[i][1].ID}_{self.bits_positive[i+1][1].ID}",
                w=1,
                d=1,
            )
            net.createSynapse(
                pre=self.bits_positive[i][1],
                post=self.bits_positive[i + 1][2],
                ID=f"{self.bits_positive[i][1].ID}_{self.bits_positive[i+1][2].ID}",
                w=1,
                d=1,
            )
        # Setup carry synapses in negative bits
        for i in range(self.negative_precision):
            net.createSynapse(
                pre=self.bits_negative[i][1],
                post=self.bits_negative[i + 1][0],
                ID=f"{self.bits_negative[i][1].ID}_{self.bits_negative[i+1][0].ID}",
                w=1,
                d=1,
            )
            net.createSynapse(
                pre=self.bits_negative[i][1],
                post=self.bits_negative[i + 1][1],
                ID=f"{self.bits_negative[i][1].ID}_{self.bits_negative[i+1][1].ID}",
                w=1,
                d=1,
            )
            net.createSynapse(
                pre=self.bits_negative[i][1],
                post=self.bits_negative[i + 1][2],
                ID=f"{self.bits_negative[i][1].ID}_{self.bits_negative[i+1][2].ID}",
                w=1,
                d=1,
            )

        # Setup synapses between positive bit neurons and positive outgoing neurons
        if self.positive_precision > 0:
            for i in range(self.positive_precision + 1):
                net.createSynapse(
                    pre=self.bits_positive[i][0],
                    post=self.z_positive[i],
                    ID=f"{self.bits_positive[i][0].ID}_{self.z_positive[i].ID}",
                    w=1,
                    d=self.higher_precision - i + 1,
                )
                net.createSynapse(
                    pre=self.bits_positive[i][1],
                    post=self.z_positive[i],
                    ID=f"{self.bits_positive[i][1].ID}_{self.z_positive[i].ID}",
                    w=-1,
                    d=self.higher_precision - i + 1,
                )

                if i > 0:
                    net.createSynapse(
                        pre=self.bits_positive[i][2],
                        post=self.z_positive[i],
                        ID=f"{self.bits_positive[i][0].ID}_{self.z_positive[i].ID}",
                        w=1,
                        d=self.higher_precision - i + 1,
                    )

        # Setup synapses between negative bit neurons and negative outgoing neurons
        if self.negative_precision > 0:
            for i in range(self.negative_precision + 1):
                net.createSynapse(
                    pre=self.bits_negative[i][0],
                    post=self.z_negative[i],
                    ID=f"{self.bits_negative[i][0].ID}_{self.z_negative[i].ID}",
                    w=1,
                    d=self.higher_precision - i + 1,
                )
                net.createSynapse(
                    pre=self.bits_negative[i][1],
                    post=self.z_negative[i],
                    ID=f"{self.bits_negative[i][1].ID}_{self.z_negative[i].ID}",
                    w=-1,
                    d=self.higher_precision - i + 1,
                )

                if i > 0:
                    net.createSynapse(
                        pre=self.bits_negative[i][2],
                        post=self.z_negative[i],
                        ID=f"{self.bits_negative[i][0].ID}_{self.z_negative[i].ID}",
                        w=1,
                        d=self.higher_precision - i + 1,
                    )
        # Synapses created
        # print("Synapses created...")
        # print("Virtual neuron created...")


def connect_virtual_neurons(A, B, C):
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
            w=1,
            d=1,
        )

    for i in range(B.negative_precision):
        net.createSynapse(
            pre=B.z_negative[i],
            post=C.y_negative[i],
            ID=f"{A.z_negative[i].ID}_{C.y_negative[i].ID}",
            w=1,
            d=1,
        )
