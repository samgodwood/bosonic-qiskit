import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit_algorithms import TimeEvolutionProblem
from qiskit_algorithms import TrotterQRTE
from qiskit.quantum_info import Statevector
from qiskit import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from scipy.sparse import csr_matrix, kron

def create_annihilation_operator(n_max):
    """
    Create the annihilation operator 'b' for a Fock space with cutoff n_max.

    Parameters:
        n_max (int): The Fock cutoff.

    Returns:
        np.ndarray: The annihilation operator matrix.
    """
    b = np.zeros((n_max + 1, n_max + 1), dtype=np.complex128)
    for n in range(1, n_max + 1):
        b[n - 1, n] = np.sqrt(n)
    return b

def create_creation_operator(n_max):
    """
    Create the creation operator 'b^dagger' for a Fock space with cutoff n_max.

    Parameters:
        n_max (int): The Fock cutoff.

    Returns:
        np.ndarray: The creation operator matrix.
    """
    return create_annihilation_operator(n_max).T.conj()



def convert_hamiltonian_to_pauli_string(hamiltonian_matrix):
    """
    Convert a Hamiltonian matrix to a Qiskit SparsePauliOp.

    Parameters:
        hamiltonian_matrix (np.ndarray): The Hamiltonian in matrix form.

    Returns:
        SparsePauliOp: The Hamiltonian as a SparsePauliOp.
    """
    return SparsePauliOp.from_operator(Operator(hamiltonian_matrix))



def evolve_system(hamiltonian, initial_state, time):
    """
    Perform time evolution on the system.

    Parameters:
        hamiltonian (SparsePauliOp): The Hamiltonian of the system.
        initial_state (Statevector): The initial quantum state.
        time (float): The total evolution time.

    Returns:
        QuantumCircuit: The evolved quantum circuit.
    """
    problem = TimeEvolutionProblem(hamiltonian, initial_state=initial_state, time=time)
    trotter = TrotterQRTE()
    result = trotter.evolve(problem)
    return result.evolved_state



def analyze_circuit(circuit, backend=GenericBackendV2(3)):
    """
    Analyze the quantum circuit to obtain gate counts and depths.

    Parameters:
        circuit (QuantumCircuit): The quantum circuit to analyze.
        backend (GenericBackendV2): The backend defining supported operations.

    Returns:
        dict: A dictionary containing gate counts, total circuit depth, and Rz gate depth.
    """
    transpiled_circuit = transpile(
        circuit,
        basis_gates=backend.operation_names,
        optimization_level=1
    )
    gate_counts = transpiled_circuit.count_ops()
    total_depth = transpiled_circuit.depth()
    rz_depth = transpiled_circuit.depth(lambda instr: instr.operation.name == 'rz')
    cnot_depth = transpiled_circuit.depth(lambda instr: instr.operation.name == 'cx')

    return {
        "gate_counts": gate_counts,
        "total_depth": total_depth,
        "rz_depth": rz_depth,
        "cnot_depth": cnot_depth
    }


def simulate_system(hamiltonian, initial_state, num_modes, n_max, additional_qubits = 0, final_time=1):
    """
    Simulate the quantum system and analyze the circuit.

    Parameters:
        hamiltonian (SparsePauliOp): The Hamiltonian of the system.
        initial_state (Statevector): The initial quantum state.
        num_modes (int): The number of modes in the system.
        n_max (int): The maximum quantum number to consider.
        final_time (float): Total evolution time. Defaults to 1.

    Returns:
        dict: Analysis results including gate counts, total depth, and Rz depth.
    """
    # Define system parameters
    n_qubits = num_modes * int(np.ceil(np.log2(n_max + 1))) + additional_qubits
    hamiltonian = convert_hamiltonian_to_pauli_string(hamiltonian)
    # Perform time evolution
    evolved_circuit = evolve_system(hamiltonian, initial_state, final_time)

    # Define a backend and analyze the circuit
    backend = GenericBackendV2(n_qubits)
    analysis_results = analyze_circuit(evolved_circuit, backend)

    return analysis_results