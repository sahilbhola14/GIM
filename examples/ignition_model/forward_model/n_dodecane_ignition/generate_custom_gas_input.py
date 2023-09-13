import yaml
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def gen_gas_file(
    reaction_1_pre_exp_factor, reaction_1_activation_energy, campaign_path=None
):
    data = {
        "description": "Custom reaction for n-dodecane combustion",
        "units": {
            "length": "cm",
            "time": "s",
            "quantity": "mol",
            "activation-energy": "cal/mol",
        },
        "phases": [
            {
                "name": "gas",
                "thermo": "ideal-gas",
                "elements": ["O", "H", "C", "N", "Ar"],
                "species": ["C12H26", "O2", "CO", "H2O", "CO2"],
                "kinetics": "gas",
                "transport": "mixture-averaged",
                "state": {"T": 300.0, "P": "1 atm"},
            }
        ],
        "species": [
            {
                "name": "O2",
                "composition": {"O": 2},
                "thermo": {
                    "model": "NASA7",
                    "temperature-ranges": [200.0, 1000.0, 3500.0],
                    "data": [
                        [
                            3.78245636,
                            -2.99673416e-03,
                            9.84730201e-06,
                            -9.68129509e-09,
                            3.24372837e-12,
                            -1063.94356,
                            3.65767573,
                        ],
                        [
                            3.28253784,
                            1.48308754e-03,
                            -7.57966669e-07,
                            2.09470555e-10,
                            -2.16717794e-14,
                            -1088.45772,
                            5.45323129,
                        ],
                    ],
                    "note": "TPIS89",
                },
                "transport": {
                    "model": "gas",
                    "geometry": "linear",
                    "well-depth": 107.4,
                    "diameter": 3.458,
                    "polarizability": 1.6,
                    "rotational-relaxation": 3.8,
                },
            },
            {
                "name": "H2O",
                "composition": {"H": 2, "O": 1},
                "thermo": {
                    "model": "NASA7",
                    "temperature-ranges": [200.0, 1000.0, 3500.0],
                    "data": [
                        [
                            4.19864056,
                            -2.0364341e-03,
                            6.52040211e-06,
                            -5.48797062e-09,
                            1.77197817e-12,
                            -3.02937267e04,
                            -0.849032208,
                        ],
                        [
                            3.03399249,
                            2.17691804e-03,
                            -1.64072518e-07,
                            -9.7041987e-11,
                            1.68200992e-14,
                            -3.00042971e04,
                            4.9667701,
                        ],
                    ],
                    "note": "L8/89",
                },
                "transport": {
                    "model": "gas",
                    "geometry": "nonlinear",
                    "well-depth": 572.4,
                    "diameter": 2.605,
                    "dipole": 1.844,
                    "rotational-relaxation": 4.0,
                },
            },
            {
                "name": "CO",
                "composition": {"C": 1, "O": 1},
                "thermo": {
                    "model": "NASA7",
                    "temperature-ranges": [200.0, 1000.0, 3500.0],
                    "data": [
                        [
                            3.57953347,
                            -6.1035368e-04,
                            1.01681433e-06,
                            9.07005884e-10,
                            -9.04424499e-13,
                            -1.4344086e04,
                            3.50840928,
                        ],
                        [
                            2.71518561,
                            2.06252743e-03,
                            -9.98825771e-07,
                            2.30053008e-10,
                            -2.03647716e-14,
                            -1.41518724e04,
                            7.81868772,
                        ],
                    ],
                    "note": "TPIS79",
                },
                "transport": {
                    "model": "gas",
                    "geometry": "linear",
                    "well-depth": 98.1,
                    "diameter": 3.65,
                    "polarizability": 1.95,
                    "rotational-relaxation": 1.8,
                },
            },
            {
                "name": "CO2",
                "composition": {"C": 1, "O": 2},
                "thermo": {
                    "model": "NASA7",
                    "temperature-ranges": [200.0, 1000.0, 3500.0],
                    "data": [
                        [
                            2.35677352,
                            8.98459677e-03,
                            -7.12356269e-06,
                            2.45919022e-09,
                            -1.43699548e-13,
                            -4.83719697e04,
                            9.90105222,
                        ],
                        [
                            3.85746029,
                            4.41437026e-03,
                            -2.21481404e-06,
                            5.23490188e-10,
                            -4.72084164e-14,
                            -4.8759166e04,
                            2.27163806,
                        ],
                    ],
                    "note": "L7/88",
                },
                "transport": {
                    "model": "gas",
                    "geometry": "linear",
                    "well-depth": 244.0,
                    "diameter": 3.763,
                    "polarizability": 2.65,
                    "rotational-relaxation": 2.1,
                },
            },
            {
                "name": "C12H26",
                "composition": {"C": 12, "H": 26},
                "thermo": {
                    "model": "NASA7",
                    "temperature-ranges": [300.0, 1391.0, 5000.0],
                    "data": [
                        [
                            -2.62182000e00,
                            1.47238000e-01,
                            -9.43970000e-05,
                            3.07441000e-08,
                            -4.03602000e-12,
                            -4.00654000e04,
                            5.00995000e01,
                        ],
                        [
                            3.85095000e01,
                            5.63550000e-02,
                            -1.91493000e-05,
                            2.96025000e-09,
                            -1.71244000e-13,
                            -5.48843000e04,
                            -1.72671000e02,
                        ],
                    ],
                },
                "transport": {
                    "model": "gas",
                    "geometry": "nonlinear",
                    "well-depth": 789.980,
                    "diameter": 7.047,
                    "polarizability": 0.0,
                    "rotational-relaxation": 1.0,
                },
            },
        ],
        "reactions": [
            {
                "equation": "C12H26 + 12.5 O2 => 12 CO + 13 H2O",
                "rate-constant": {
                    "A": reaction_1_pre_exp_factor,
                    "b": 0.0,
                    "Ea": reaction_1_activation_energy,
                },
                "orders": {"C12H26": 0.25, "O2": 1.25},
            },
            {
                "equation": "CO + 0.5 O2 => CO2",
                "rate-constant": {
                    "A": 3.98e14,
                    "b": 0.0,
                    "Ea": 40000.0,
                },
                "orders": {"CO": 1.0, "H2O": 0.5, "O2": 0.25},
                "nonreactant-orders": True,
            },
            {
                "equation": "CO2 => CO + 0.5 O2",
                "rate-constant": {
                    "A": 5.0e8,
                    "b": 0.0,
                    "Ea": 40000.0,
                },
                "orders": {
                    "CO2": 1.0,
                },
            },
        ],
    }
    if campaign_path is not None:
        file_path = os.path.join(
            campaign_path, "custom_gas_rank_" + str(rank) + ".yaml"
        )
    else:
        file_path = "./custom_gas_rank_" + str(rank) + ".yaml"

    with open(file_path, "w") as file:
        yaml.dump(data, file, sort_keys=False, default_flow_style=False)


def main():
    gen_gas_file(
        reaction_1_pre_exp_factor=24464314506.42409,
        reaction_1_activation_energy=31944.0,
    )


if __name__ == "__main__":
    main()
