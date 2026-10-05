"""Short, executable interfaces printed in the manuscript's code appendix.

The full algorithms live in the case modules. These examples expose important
arguments and conventions without presenting a notebook as a required format.
"""
import numpy as np
import pennylane as qml
from sklearn.svm import SVC
from portfolio import encode, qaoa_probabilities
from pricing_risk import amplitude_matrix
from quantum_learning import states, fidelity
from teaching_cases import pqc_object_sizes
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def local_register_probabilities(n=8):
    # This is a CPU simulator: wires are labels, not remotely reserved qubits.
    device = qml.device("default.qubit", wires=n, shots=None)

    @qml.qnode(device)
    def circuit():
        # The initial register is |0...0>; Hadamards prepare a uniform state.
        for wire in range(n):
            qml.Hadamard(wires=wire)
        return qml.probs(wires=range(n))

    probabilities = np.asarray(circuit())
    np.testing.assert_allclose(probabilities, np.full(2**n, 1 / 2**n))
    return probabilities


def bit_order_example():
    # Prepare qubit 0 = 1, qubit 1 = 0, identically in both SDKs.
    device = qml.device("default.qubit", wires=2)

    @qml.qnode(device)
    def circuit():
        qml.PauliX(wires=0)
        return qml.probs(wires=[0, 1])

    qc = QuantumCircuit(2)
    qc.x(0)
    pl = np.asarray(circuit())
    qk = Statevector.from_instruction(qc).probabilities()
    # Swap tensor axes: Qiskit's qubit 0 is its least significant index bit.
    aligned = qk.reshape(2, 2).transpose(1, 0).ravel()
    np.testing.assert_allclose(pl, aligned)
    return {"PennyLane": pl, "Qiskit_raw": qk, "Qiskit_aligned": aligned}


def rotation_probability(angle=np.pi / 3):
    # A simulator evaluates the same Born probability as the ideal circuit.
    device = qml.device("default.qubit", wires=1)

    @qml.qnode(device)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.probs(wires=[0])

    probabilities = np.asarray(circuit(angle))
    assert np.isclose(probabilities[1], np.sin(angle / 2) ** 2)
    return probabilities


def portfolio_distribution(mu, covariance, layers=2):
    # Four equal-weight holdings: the QUBO coefficients include this scaling.
    bits, base, cost, h, j, penalty = encode(mu, covariance, k=4)
    # Each layer has (gamma, beta), not one angle for every stock.
    angles = np.tile([0.2, 0.1], layers)
    probabilities = qaoa_probabilities(angles, h, j)
    assert np.isclose(probabilities.sum(), 1.0)
    # Fixed angles illustrate the interface; optimisation is a separate step.
    return probabilities


def payoff_expectation(probability, payoff):
    # Bounded non-negative payoffs determine controlled ancilla rotations.
    cap = np.max(payoff)
    if cap == 0:
        return 0.0
    preparation = amplitude_matrix(probability, payoff / cap)
    # Ancilla is the final wire: odd indices have its bit equal to one.
    amplitude = np.sum(np.abs(preparation[1::2, 0]) ** 2)
    assert np.isclose(cap * amplitude, probability @ payoff)
    # Discount outside this helper, exactly once, for a monetary price.
    return cap * amplitude


def kernel_scores(x_fit, y_fit, x_test, scale=1.0, layers=1, c=10):
    # Inputs have already been scaled with training-only statistics.
    train_states = states(x_fit, layers=layers, scale=scale)
    test_states = states(x_test, layers=layers, scale=scale)
    train_kernel = fidelity(train_states, train_states)
    test_kernel = fidelity(test_states, train_states)
    # Prediction columns must have exactly the training-observation order.
    assert test_kernel.shape == (len(x_test), len(x_fit))
    model = SVC(C=c, kernel="precomputed").fit(train_kernel, y_fit)
    return model.decision_function(test_kernel)


def spectral_features(sampled_train, sampled_cross):
    # Symmetry alone does not imply a positive-semidefinite measured kernel.
    symmetric = (sampled_train + sampled_train.T) / 2
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    keep = eigenvalues > max(1e-3 * eigenvalues.max(), 1e-9)
    lam, basis = eigenvalues[keep], vectors[:, keep]
    # Define the test map too; do not repair only the training interface.
    fit = basis * np.sqrt(lam)
    test = sampled_cross @ basis / np.sqrt(lam)
    return fit, test


def security_size_example():
    # Standard-derived object lengths; no encryption or signing is executed.
    sizes = pqc_object_sizes()
    total = sizes["kem_key_and_ciphertext"] + sizes["dsa_key_and_signature"]
    assert total == 7533
    return {"four_objects_bytes": total,
            "million_signatures_bytes": sizes["one_million_signatures_bytes"]}


def smoke_test():
    local_register_probabilities()
    bit_order_example()
    rotation_probability()
    mu = np.linspace(0.01, 0.1, 8)
    portfolio_distribution(mu, np.eye(8) * 0.04)
    payoff_expectation(np.array([0.1, 0.2, 0.3, 0.4]), np.arange(4.0))
    rng = np.random.default_rng(11)
    x = rng.normal(size=(12, 4))
    scores = kernel_scores(x[:8], np.arange(8) % 2, x[8:])
    assert scores.shape == (4,)
    k = fidelity(states(x[:8], 1), states(x[:8], 1))
    cross = fidelity(states(x[8:], 1), states(x[:8], 1))
    fit, test = spectral_features(k, cross)
    assert fit.shape[1] == test.shape[1] > 0
    security_size_example()
    return "All eight printed interfaces executed successfully."


if __name__ == "__main__":
    print(smoke_test())
