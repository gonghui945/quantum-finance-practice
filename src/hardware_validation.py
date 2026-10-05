"""Cross-SDK parity, compilation and finite-shot/noise sensitivity studies.

All backends here are local. The line coupling graph is illustrative, not a
named processor or a claim of experimental access to physical qubits.
"""
from itertools import product
import importlib.metadata
import json
import platform
from time import perf_counter
import numpy as np
import pandas as pd
import pennylane as qml
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from common import RESULTS, save_json, save_figure, plt
from portfolio import qaoa_probabilities
from quantum_learning import feature_state, fidelity, states


def qiskit_qaoa(theta,h,j):
    n=len(h);qc=QuantumCircuit(n)
    qc.h(range(n))
    for gamma,beta in np.asarray(theta).reshape(-1,2):
        for i in range(n): qc.rz(2*gamma*h[i],i)
        for i in range(n):
            for k in range(i+1,n):
                if abs(j[i,k])>1e-14: qc.rzz(2*gamma*j[i,k],i,k)
        for i in range(n): qc.rx(2*beta,i)
    return qc


def parity_and_compilation():
    parameters=json.loads((RESULTS/"portfolio_parameters.json").read_text())
    rows=[];max_error=0.
    for depth in [1,2]:
        config=next(x for x in parameters if x["month"]==8 and x["depth"]==depth and x["seed"]==11)
        h,j,theta=np.array(config["h"]),np.array(config["j"]),config["theta"]
        for n in sorted({4,6,8,len(h)}):
            # Cropped Hamiltonians isolate compilation costs, not new finance universes.
            qc=qiskit_qaoa(theta,h[:n],j[:n,:n])
            qprob=Statevector.from_instruction(qc).probabilities().reshape([2]*n).transpose(list(reversed(range(n)))).reshape(-1)
            pl=qaoa_probabilities(theta,h[:n],j[:n,:n])
            err=float(np.max(abs(qprob-pl)));assert err<1e-11;max_error=max(max_error,err)
            for topology in ["All-to-all","Line"]:
                coupling=None if topology=="All-to-all" else CouplingMap.from_line(n)
                compiled=transpile(qc,basis_gates=["rz","sx","x","cx"],coupling_map=coupling,
                                   optimization_level=2,seed_transpiler=11)
                rows.append(dict(n=n,depth=depth,topology=topology,logical_depth=qc.depth(),
                                 compiled_depth=compiled.depth(),cx=compiled.count_ops().get("cx",0),
                                 all_gates=compiled.size(),probability_error=err))
    pd.DataFrame(rows).to_csv(RESULTS/"hardware_compilation.csv",index=False)
    return max_error


def noisy_kernel(x,y,layers,noise,cancel_terminal=False):
    n=len(x);dev=qml.device("default.mixed",wires=n)
    def ops(v):
        out=[]
        for _ in range(layers):
            for i in range(n): out.extend([("ry",i,float(v[i])),("rz",i,float(v[(i+1)%n]))])
            out.extend([("cx",i,i+1) for i in range(n-1)])
        # The last fixed entangler cancels against its inverse in an overlap.
        # Removing it before adding noise models an algebraic optimisation.
        return out[:-(n-1)] if cancel_terminal else out
    @qml.qnode(dev)
    def circuit():
        operations=[(o,False) for o in ops(x)]+[(o,True) for o in reversed(ops(y))]
        for (kind,i,arg),inverse in operations:
            if kind=="cx":
                qml.CNOT(wires=[i,arg])
                if noise:
                    qml.DepolarizingChannel(noise,wires=i);qml.DepolarizingChannel(noise,wires=arg)
            else:
                (qml.RY if kind=="ry" else qml.RZ)((-1 if inverse else 1)*arg,wires=i)
                if noise: qml.DepolarizingChannel(noise,wires=i)
        return qml.probs(wires=range(n))
    return float(circuit()[0])


def run():
    parity=parity_and_compilation()
    arrays=np.load(RESULTS/"qml_kernel_inputs.npz")
    kf,kx,yf,yt=arrays["kf"],arrays["kx"],arrays["yf"],arrays["yt"]
    summary=json.loads((RESULTS/"qml_summary.json").read_text())
    config=next(c for c in summary["selected"] if c["method"]=="Reuploading kernel")
    rows=[]
    for shots in [128,512,2048]:
        for seed in range(10):
            rng=np.random.default_rng(seed)
            sampled=rng.binomial(shots,np.clip(kf,0,1))/shots
            sampled=np.triu(sampled,1);sampled=sampled+sampled.T+np.eye(len(kf))
            cross=rng.binomial(shots,np.clip(kx,0,1))/shots
            eigen,vectors=np.linalg.eigh(sampled)
            # A landmark feature projection gives a coherent out-of-sample map.
            keep=eigen>max(1e-3*eigen.max(),1e-9)
            v=vectors[:,keep];lam=eigen[keep]
            training_features=v*np.sqrt(lam)
            test_features=cross@v/np.sqrt(lam)
            model=SVC(C=config["C"],kernel="linear").fit(training_features,yf)
            score=model.decision_function(test_features)
            rows.append(dict(shots=shots,seed=seed,negative_eigenvalues=int((eigen<-1e-10).sum()),
                             retained_rank=int(keep.sum()),auc=roc_auc_score(yt,score),
                             kernel_rmse=float(np.sqrt(np.mean((sampled-kf)**2)))))
    pd.DataFrame(rows).to_csv(RESULTS/"hardware_shot_kernel.csv",index=False)
    # Hold 24 feature pairs fixed to separate channel noise from market sampling.
    x=arrays["xf"][:24];y=arrays["xt"][:24];noise_rows=[]
    for depth in [1,2]:
        exact=np.array([abs(np.vdot(feature_state(a,depth),feature_state(b,depth)))**2 for a,b in zip(x,y)])
        for cancelled in [False,True]:
            for eta in [0.,.001,.005,.01]:
                values=np.array([noisy_kernel(a,b,depth,eta,cancelled) for a,b in zip(x,y)])
                if eta==0: assert np.max(abs(values-exact))<1e-10
                noise_rows.append(dict(depth=depth,cancelled=cancelled,eta=eta,pairs=24,mean_kernel=float(values.mean()),
                                       rmse=float(np.sqrt(np.mean((values-exact)**2))),max_error=float(np.max(abs(values-exact)))))
    pd.DataFrame(noise_rows).to_csv(RESULTS/"hardware_noise_kernel.csv",index=False)
    env={k:importlib.metadata.version(k) for k in ["numpy","pandas","scipy","scikit-learn","pennylane","qiskit","matplotlib","nbformat","nbclient"]}
    env.update(python=platform.python_version(),machine=platform.machine(),platform=platform.system())
    save_json("environment.json",env)
    result=dict(max_sdk_probability_error=parity,shot_summary=pd.DataFrame(rows).groupby("shots").agg(
        auc_mean=("auc","mean"),auc_std=("auc","std"),auc_min=("auc","min"),auc_max=("auc","max"),
        negative_mean=("negative_eigenvalues","mean"),rank_mean=("retained_rank","mean"),kernel_rmse=("kernel_rmse","mean")).reset_index().to_dict("records"),
        noise=noise_rows,train=len(yf),test=len(yt),environment=env)
    save_json("hardware_summary.json",result)
    shot=pd.DataFrame(rows);noise=pd.DataFrame(noise_rows)
    fig,ax=plt.subplots(1,2,figsize=(7,3.4))
    group=shot.groupby("shots").auc.agg(["mean","std"])
    ax[0].errorbar(group.index,group["mean"],yerr=group["std"],marker="o",capsize=3)
    ax[0].set_xscale("log",base=2);ax[0].set(xlabel="Shots per kernel entry",ylabel="AUC (mean +/- seed SD)")
    ax[0].axhline(next(x["auc"] for x in summary["metrics"] if x["method"]=="Reuploading kernel"),ls="--",color="black",label="Exact kernel")
    ax[0].legend(fontsize=8)
    for (depth,cancelled),g in noise.groupby(["depth","cancelled"]):
        ax[1].plot(g.eta,g.rmse,marker="o",ls="--" if cancelled else "-",label=f"L={depth}, {'reduced' if cancelled else 'raw'}")
    ax[1].set(xlabel="Depolarising probability per gate/wire",ylabel="Kernel RMSE on 24 fixed pairs");ax[1].legend()
    save_figure("hardware_sensitivity")
    return result


if __name__=="__main__": print(run())
