import sys
import os

# Adjust the path based on your directory structure
module_path = os.path.abspath(os.path.join('..', '..', '..'))  # Moves three directories up
if module_path not in sys.path:
    sys.path.append(module_path)
    
# Now you can import c2qa and other modules from bosonic-qiskit
import c2qa
import qiskit
import numpy as np
import c2qa.util as util
def hopping_term(circuit, qbr, qb_index, qmr, mode1, mode2, j, delta_t):

    """
    Applies the hopping term \( \hat{U}_2 \) as a conditional beam splitter.

    Parameters:
        circuit (QuantumCircuit): The quantum circuit to modify.
        qbr (QuantumRegister): The quantum register containing the gauge field qubits.
        qb_index (int): The index of the qubit in the quantum register acting as the control.
        qmr (QumodeRegister): The quantum register containing the bosonic modes.
        mode1 (int): The index of the first bosonic mode.
        mode2 (int): The index of the second bosonic mode.
        j (float): Coupling constant for the hopping term.
        delta_t (float): Trotter time step.
    """
    theta = -1j * j * delta_t
    circuit.cv_c_bs(theta, qmr[mode1], qmr[mode2], qbr[qb_index])

def e_field(circuit, qbr, qb_index, g, delta_t):
    """
    Applies the electric field term \( \hat{U}_1 \) to a specific qubit in the gauge field qubit register.

    Parameters:
        circuit (QuantumCircuit): The quantum circuit to modify.
        qbr (QuantumRegister): The quantum register containing the gauge field qubits.
        qb_index (int): The index of the qubit in the quantum register.
        g (float): Coupling constant for the electric field term.
        delta_t (float): Trotter time step.
    """
    theta = 2 * g * delta_t
    circuit.rx(theta, qbr[qb_index])