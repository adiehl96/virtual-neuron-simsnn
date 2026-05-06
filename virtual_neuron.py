from simsnn.core.networks import Network
from simsnn.core.simulators import Simulator
from pprint import pprint
import numpy as np
from snnsimsnn.func import regexget


def run(a, b, duration=10, plotting=False):

    net = Network()
    sim = Simulator(net)

    lbr = len(format(a, "b"))  # Length of the binary representation
    lbr = len(format(b, "b")) if len(format(b, "b")) > lbr else lbr

    print(lbr)

    nrn = {f"x_{i}": net.createLIF(ID=f"x_{i}") for i in range(lbr)}
    nrn = nrn | {f"y_{i}": net.createLIF(ID=f"y_{i}") for i in range(lbr)}
    nrn = nrn | {
        f"I_{i}_{j}": net.createLIF(ID=f"I_{i}_{j}", thr=j + 1)
        for i in range(lbr + 1)
        for j in range(3)
        if not (i == 0 and j == 2)
    }

    nrn = nrn | {f"o_{i}": net.createLIF(ID=f"o_{i}") for i in range(lbr + 1)}

    syn = {}

    # Input to Interneuron connections
    for inp in ["x", "y"]:
        for i in range(lbr):
            for psn in regexget(nrn, f"I_{i}_."):
                syn = syn | {
                    f"{inp}_{i}_{psn.ID}": net.createSynapse(
                        pre=nrn[f"{inp}_{i}"],
                        post=psn,
                        ID=f"{inp}_{i}_{psn.ID}",
                        w=1,
                        d=i + 1,
                    )
                }

    # Inter Interneuron connections
    for group in range(lbr):
        for psn in regexget(nrn, f"I_{group+1}_."):
            syn = syn | {
                f"I_{group}_1_{psn.ID}": net.createSynapse(
                    pre=nrn[f"I_{group}_1"],
                    post=psn,
                    ID=f"I_{group}_1_{psn.ID}",
                    w=1,
                    d=1,
                )
            }

    # Interneuron Output connections
    for group in range(lbr + 1):
        delay = len(regexget(nrn, f"o_.")) - group
        for psn in regexget(nrn, f"I_{group}_."):
            syn = syn | {
                f"{psn.ID}_o_{group}": net.createSynapse(
                    pre=nrn[f"{psn.ID}"],
                    post=nrn[f"o_{group}"],
                    ID=f"{psn.ID}_o_{group}",
                    w=-1 if psn.ID.endswith("1") else 1,
                    d=delay,
                )
            }

    encode = lambda x: format(x, "b").zfill(lbr)[::-1]
    x_encoded = encode(a)
    y_encoded = encode(b)
    inputs = {"x_encoded": x_encoded, "y_encoded": y_encoded}
    for inp in ["x", "y"]:
        for idx, i in enumerate(inputs[f"{inp}_encoded"]):
            nrn = nrn | {
                f"inp_{inp}_{idx}": net.createInputTrain(
                    train=[i], loop=False, ID=f"inp_{inp}_{idx}"
                )
            }

    for inp in ["x", "y"]:
        for idx in range(len(inputs[f"{inp}_encoded"])):
            syn = syn | {
                f"inp_{inp}_{idx}_{inp}_{idx}": net.createSynapse(
                    pre=nrn[f"inp_{inp}_{idx}"],
                    post=nrn[f"{inp}_{idx}"],
                    ID=f"inp_{inp}_{idx}_{inp}_{idx}",
                    w=1,
                    d=1,
                )
            }

    # Add all neurons to the raster
    sim.raster.addTarget(regexget(nrn, "x_.|y_.|o_."))
    # Add all neurons to the multimeter
    # sim.multimeter.addTarget(regexget(nrn, "A.*"))

    sim.run(duration, plotting=plotting)

    # read out and convert output
    rasterdata = sim.raster.get_measurements()

    # readout time is semi hardcoded
    read_out_time = 2 + lbr + 1
    n_add_rasterdata = rasterdata[read_out_time][-(lbr + 1) :][::-1]
    n_add_rasterdata = "".join(list(map(lambda x: str(int(x)), n_add_rasterdata)))
    output = int(n_add_rasterdata, 2)
    return output
