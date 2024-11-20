import sys
import os

# Adjust the path based on your directory structure
module_path = os.path.abspath(os.path.join('..', '..'))  # Moves two directories up
if module_path not in sys.path:
    sys.path.append(module_path)

# Now you can import c2qa and other modules from bosonic-qiskit
import c2qa
import qiskit
import numpy as np
# Math, numerics, and graphing
import numpy as np
import scipy as sp
import scipy.integrate as integrate
from scipy.optimize import minimize
from scipy.special import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from scipy.integrate import trapz
from scipy.special import eval_hermite


def fock_to_position(n, q):
    '''Returns the Fock position wavefunctions defined over q space.

    Args:
        n (int): Fock index
        q (array): position values

    Returns:
        position_wf_fock (array): nth Fock state wavefunction
    '''

    position_wf_fock = ((1/np.sqrt((2**n)*factorial(n)))*(np.pi**-0.25)
            *np.exp(-(q**2)/2)*eval_hermite(n,q) ) #Employs the scipy.special eval_hermite function

    return position_wf_fock

def fock_coefficients_to_position(coeffs, q):
    '''Returns the position wavefunction for a superposition of Fock states
    defined by the given Fock coefficients.

    Args:
        coeffs (array): array of Fock coefficients
        q (array): position values over which to compute the wavefunction

    Returns:
        wavefunction (array): resulting wavefunction in the position basis
    '''
    wavefunction = np.zeros_like(q, dtype=complex)  # Initialize the wavefunction

    # Loop over all Fock coefficients and sum the weighted Fock states
    for n, coeff in enumerate(coeffs):
        fock_wf = fock_to_position(n, q)  # Get the nth Fock state in position basis
        wavefunction += coeff * fock_wf   # Add the weighted Fock state to the total wavefunction

    return wavefunction

def fock_density_matrix_to_position(fock_density_matrix, q):
    '''Convert Fock basis density matrix to position-space density matrix.

    Args:
        fock_density_matrix (2D array): Fock basis density matrix (rho_mn)
        q (array): Position values for the grid

    Returns:
        position_density_matrix (2D array): Density matrix in position space
    '''
    N = fock_density_matrix.shape[0]  # Assuming square matrix of size N x N, which will always be true in Bosonic qiskit
    position_density_matrix = np.zeros((len(q), len(q)), dtype=complex)
    
    # Loop over all matrix elements rho_mn
    for m in range(N):
        for n in range(N):
            # Get the Fock wavefunctions in position basis
            psi_m_x = fock_to_position(m, q)
            psi_n_x_prime = fock_to_position(n, q)
            
            # Add the weighted outer product of wavefunctions to the position density matrix
            position_density_matrix += fock_density_matrix[m, n] * np.outer(psi_m_x, psi_n_x_prime.conj())
    
    return position_density_matrix

def gkp_fock_coeff(n, combs=100, alpha=np.sqrt(np.pi), mu=0, aspect_ratio=1):
    '''Returns the nth Fock coefficient for the ideal logical GKP state.

    Args:
        n (int): index of Fock state
        combs (int): number of delta spikes in the ideal state used to 
                        calculate inner product with Fock state
        alpha (float): GKP lattice constant
        mu (0 or 1): logical value of GKP state
        aspect_ratio (float): the aspect ratio lambda that scales position and momentum lattice spacing

    Returns:
        coeff (complex): inner product between ideal GKP and nth 
            Fock states.
    '''
    # adjust alpha by aspect_ratio in the q direction
    alpha_q = alpha * aspect_ratio
    # q values from which to sample the nth Fock state
    samples = (mu + np.arange(-combs/2, 1+combs/2)*2)*alpha_q
    # sum of sampled values to yield inner product
    coeff = np.sum(fock_to_position(n, samples))
    return coeff

def gkp_fock(qubit, eps, q, n_max=100, norm=True, aspect_ratio=1): 
    '''Returns the epsilon GKP q wavefunction and coefficients in the 
        Fock basis.

    Args:
        qubit (array): qubit state in logical basis
        eps (float): value of epsilon
        q (array): array of q values for q wavefunction
        n_max (int): Fock cutoff
        norm (Boolean): whether to return a normalized state
        aspect_ratio (float): the aspect ratio lambda that scales position and momentum lattice spacing

    Returns:
        gkp (array): q wavefunction
        coeffs (array): Fock coefficients up to cutoff.
    '''
    qubit = qubit/np.linalg.norm(qubit)
    gkp = 0
    coeffs = np.zeros(n_max+1, dtype=complex)
    # initialize normalization constant
    N = 0
    for i in range(n_max+1):
        # calculate nth coefficient and weight it with an epsilon dependent exponential
        coeff = (qubit[0]*gkp_fock_coeff(i, mu=0, aspect_ratio=aspect_ratio) + qubit[1]*gkp_fock_coeff(i, mu=1, aspect_ratio=aspect_ratio))*np.exp(-i*eps)
        coeffs[i] = coeff
        gkp += coeff*fock_to_position(i, q)
        if norm:
            N += np.absolute(coeff)**2
    if norm:
        gkp = gkp/np.sqrt(N)
        coeffs = coeffs/np.sqrt(N)
    return gkp, coeffs

def qunaught_fock_coeff(n, combs=100, alpha=np.sqrt(2*np.pi), aspect_ratio=1):
    '''Returns the nth Fock coefficient for the ideal qunaught GKP state.

    Args:
        n (int): index of Fock state
        combs (int): number of delta spikes in the ideal state used to 
                        calculate inner product with Fock state
        alpha (float): Qunaught lattice constant (default is sqrt(2*pi))
        aspect_ratio (float): the aspect ratio lambda that scales position and momentum lattice spacing

    Returns:
        coeff (complex): inner product between ideal qunaught GKP and nth 
            Fock states.
    '''
    # adjust alpha by aspect_ratio in the q direction
    alpha_q = alpha * aspect_ratio
    # q values from which to sample the nth Fock state
    samples = (np.arange(-combs/2, 1+combs/2))*alpha_q
    # sum of sampled values to yield inner product
    coeff = np.sum(fock_to_position(n, samples))
    return coeff


def qunaught_fock(eps, q, n_max=100, norm=True, aspect_ratio=1): 
    '''Returns the epsilon-qunaught q wavefunction and coefficients in the 
        Fock basis.

    Args:
        eps (float): squeezing parameter (epsilon) for the qunaught state
        q (array): array of q values for q wavefunction
        n_max (int): Fock cutoff (maximum number of Fock states considered)
        norm (Boolean): whether to return a normalized state
        aspect_ratio (float): the aspect ratio lambda that scales position and momentum lattice spacing

    Returns:
        qunaught (array): q wavefunction for qunaught state
        coeffs (array): Fock coefficients up to cutoff.
    '''
    alpha = np.sqrt(2*np.pi)  # Lattice constant for qunaught is sqrt(2*pi)

    qunaught = 0
    coeffs = np.zeros(n_max+1, dtype=complex)
    # initialize normalization constant
    N = 0
    for i in range(n_max+1):
        # calculate nth coefficient for the qunaught state, with aspect_ratio
        coeff = qunaught_fock_coeff(i, alpha=alpha, aspect_ratio=aspect_ratio) * np.exp(-i * eps)
        coeffs[i] = coeff
        qunaught += coeff * fock_to_position(i, q)
        if norm:
            N += np.absolute(coeff)**2
    
    # Check if normalization constant is non-zero before dividing
    if norm and N > 0:
        qunaught = qunaught / np.sqrt(N)
        coeffs = coeffs / np.sqrt(N)
    elif N == 0:
        print("Warning: Normalization constant is zero, skipping normalization.")
        
    return qunaught, coeffs

from qutip import *

def compute_Delta_eff_dB_q(fock_density_matrix):
    '''
    Compute Δ_eff^{dB} based on Tr[S_q ρ], where S_q = e^{i sqrt(2π) q_op} and ρ is the density matrix in the Fock basis.

    Args:
        fock_density_matrix (Qobj or ndarray): Fock basis density matrix (rho_mn)

    Returns:
        Delta_eff_dB (float): The value of Δ_eff^{dB}
    '''
    # Ensure fock_density_matrix is a Qobj
    if not isinstance(fock_density_matrix, Qobj):
        fock_density_matrix = Qobj(fock_density_matrix)

    # Get the dimension N from the density matrix
    N = fock_density_matrix.shape[0]

    # Parameters
    alpha = 1j * np.sqrt(2 * np.pi)/np.sqrt(2)  # Displacement parameter

    # Define the displacement operator D_alpha
    D_alpha = displace(N, alpha)

    # Compute Tr[S_q ρ] = Tr[D_alpha * rho]
    
    Tr_Sq_rho =  (D_alpha * fock_density_matrix).tr()

    # Compute |Tr[S_q ρ]|
    abs_Tr_Sq_rho = np.abs(Tr_Sq_rho)

    # Compute the natural logarithm ln(1/|Tr[S_q ρ]|)
    ln_one_over_abs_Tr_Sq_rho = np.log(1 / abs_Tr_Sq_rho)

    # Compute Δ_eff^{dB}
    Delta_eff_dB = 10 * np.log10(0.5 / (ln_one_over_abs_Tr_Sq_rho / np.pi))

    return Delta_eff_dB


# Function to apply the Big-Small-Big sequence
def apply_bsb_sequence(circuit, q_mode, qbit, ell, eps, theta):
    circuit.h(qbit)  # Initialize qubit in |+> state after reset
    circuit.cv_c_d(ell, q_mode, qbit) # First Big Operation (Controlled Displacement
    circuit.rx(theta, qbit)   # Apply Rx rotation
    circuit.cv_c_d(2 * eps, q_mode, qbit)     # Small Operation (Controlled Displacement)
    circuit.rx(-theta, qbit)     # Apply Rx dagger (inverse of Rx)
    circuit.cv_c_d(ell, q_mode, qbit) # Second Big Operation (Controlled Displacement)
    circuit.reset(qbit) # Reset qubit to complete the cycle

# Function to apply the sBs sequence for q and p
def apply_sbs_sequence(circuit, q_mode, qbit, ell, eps, theta):
    circuit.h(qbit)  ## Initialize qubit in the |+> state
    circuit.cv_c_d(eps / 2, q_mode, qbit)  ## SMALL operation
    circuit.rx(-theta, qbit)  ## Qubit rotation Rx(-theta)
    circuit.cv_c_d(ell, q_mode, qbit)  ## BIG operation
    circuit.rx(theta, qbit)  ## Qubit rotation Rx(theta)
    circuit.cv_c_d(eps / 2, q_mode, qbit)  ## SMALL operation
    circuit.reset(qbit)  ## Reset the qubit to complete the cycle

# Function to apply the sharpen sequence for q and p
def apply_sharpen_sequence(circuit, q_mode, qbit, ell, eps, theta):
    circuit.h(qbit)  ## Initialize qubit in the |+> state
    circuit.cv_c_d(ell, q_mode, qbit)  ## big operation
    circuit.rx(theta, qbit)
    circuit.cv_c_d(eps, q_mode, qbit)  ## SMALL operation
    circuit.reset(qbit)

def apply_trim_sequence(circuit, q_mode, qbit, ell, eps, theta):
    circuit.h(qbit)  ## Initialize qubit in the |+> state
    circuit.cv_c_d(eps, q_mode, qbit)  ## SMALL operation
    circuit.rx(-theta, qbit) #rx dagger
    circuit.cv_c_d(ell, q_mode, qbit)  ## big operation
    circuit.reset(qbit)

def run__qunaught_prep_circuit(method_name, rounds, ell_q, eps_q, ell_p, eps_p, theta, num_qubits_per_mode):
    """
    Creates and simulates a quantum circuit based on the specified method and parameters.

    Parameters:
        method_name (str): The name of the method ('BsB', 'sBs', or 'Sharpen Trim').
        rounds (int): The number of rounds to apply the sequence.
        ell_q, eps_q, ell_p, eps_p, theta: Parameters for the sequences.
        num_qubits_per_mode (int): Number of qubits per qumode (determines Fock cutoff).

    Returns:
        float: The computed Delta_eff_dB for the method.
    """
    # Create qumodes and qubits
    qmr = c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=num_qubits_per_mode)
    qbr = qiskit.QuantumRegister(1)
    cr = qiskit.ClassicalRegister(1) if method_name == 'BsB' else None
    circuit = c2qa.CVCircuit(qmr, qbr, cr) if cr else c2qa.CVCircuit(qmr, qbr)
    
    # Apply sequences based on method
    if method_name == 'BsB':
        for _ in range(rounds):
            apply_bsb_sequence(circuit, qmr[0], qbr[0], ell_q, eps_q, theta)
            apply_bsb_sequence(circuit, qmr[0], qbr[0], ell_p, eps_p, theta)
    elif method_name == 'sBs':
        for _ in range(rounds):
            for _ in range(2):
                apply_sbs_sequence(circuit, qmr[0], qbr[0], ell_q, eps_q, theta)
            for _ in range(2):
                apply_sbs_sequence(circuit, qmr[0], qbr[0], ell_p, eps_p, theta)
    elif method_name == 'Sharpen Trim':
        for _ in range(rounds):
            apply_sharpen_sequence(circuit, qmr[0], qbr[0], ell_q, eps_q, theta)
            apply_trim_sequence(circuit, qmr[0], qbr[0], ell_q, eps_q, theta)
            apply_sharpen_sequence(circuit, qmr[0], qbr[0], ell_p, eps_p, theta)
            apply_trim_sequence(circuit, qmr[0], qbr[0], ell_p, eps_p, theta)
    else:
        raise ValueError(f"Unknown method {method_name}")
    
    # Simulate the circuit
    state, _, _ = c2qa.wigner.simulate(circuit)
    density_matrix = np.array(c2qa.util.trace_out_qubits(circuit, state))
    Delta_eff = compute_Delta_eff_dB_q(density_matrix)
    return Delta_eff