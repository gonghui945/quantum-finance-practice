"""Eight-asset equal-weight selection: exact, swap and simulated QAOA.

Analytic simulator optimisation is intentionally distinguished from the final
finite-shot decoding. This is a circuit/decision experiment, not QPU timing.
"""
from itertools import product
from math import comb
from time import perf_counter
import numpy as np
import pandas as pd
import pennylane as qml
from scipy.optimize import minimize
from common import TICKERS, PORTFOLIO_K, RESULTS, market, close_panel, save_json, save_figure, plt


def encode(mu, cov, k=PORTFOLIO_K, risk=5.):
    """Return a QUBO with a provably sufficient penalty on this small domain."""
    n = len(mu)
    bits = np.array(list(product([0, 1], repeat=n)), dtype=int)
    base = risk * np.einsum("bi,ij,bj->b", bits, cov, bits) / k**2 - bits @ mu / k
    # A PSD covariance makes the quadratic part non-negative. Its range is at
    # most risk*sum(abs(cov))/k^2; the linear range is at most sum(abs(mu))/k.
    penalty = float(risk*np.abs(cov).sum()/k**2 + np.abs(mu).sum()/k + .01)
    linear = risk * np.diag(cov) / k**2 - mu / k + penalty * (1 - 2*k)
    quadratic = np.triu(2*risk*cov/k**2 + 2*penalty, 1)
    cost = base + penalty * (bits.sum(axis=1) - k)**2
    reconstructed = penalty*k*k + bits @ linear + np.einsum("bi,ij,bj->b", bits, quadratic, bits)
    assert np.max(np.abs(cost-reconstructed)) < 1e-10
    # z_i=1-2*x_i. Keep the constant separately to test every basis state.
    h = -.5*linear - .25*(quadratic.sum(axis=0)+quadratic.sum(axis=1))
    j = quadratic/4
    const = penalty*k*k + .5*linear.sum() + .25*quadratic.sum()
    z = 1-2*bits
    assert np.max(np.abs(cost-(const+z@h+np.einsum("bi,ij,bj->b",z,j,z)))) < 1e-10
    scale = max(np.max(np.abs(h)), np.max(np.abs(j)), 1e-9)
    return bits, base, cost, h/scale, j/scale, penalty


def qaoa_probabilities(theta, h, j, noise=0.):
    """Gate implementation usable in both pure- and mixed-state experiments."""
    n = len(h)
    device = qml.device("default.mixed" if noise else "default.qubit", wires=n)
    @qml.qnode(device)
    def circuit():
        for i in range(n): qml.Hadamard(wires=i)
        for gamma, beta in np.asarray(theta).reshape(-1,2):
            for i in range(n):
                qml.RZ(2*gamma*h[i], wires=i)
            for i in range(n):
                for k in range(i+1,n):
                    if abs(j[i,k]) > 1e-14:
                        qml.IsingZZ(2*gamma*j[i,k], wires=[i,k])
                        if noise:
                            qml.DepolarizingChannel(noise,wires=i)
                            qml.DepolarizingChannel(noise,wires=k)
            for i in range(n): qml.RX(2*beta,wires=i)
        return qml.probs(wires=range(n))
    return np.asarray(circuit())


def local_swap(base, bits, mu, k):
    selected = set(np.argsort(mu)[-k:])
    lookup = {tuple(row):i for i,row in enumerate(bits)}
    def index(s): return lookup[tuple(int(i in s) for i in range(len(mu)))]
    evaluations = 1
    while True:
        old = index(selected)
        candidates = [selected-{i}|{j} for i in sorted(selected) for j in range(len(mu)) if j not in selected]
        values = [(base[index(s)],sorted(s),s) for s in candidates]
        evaluations += len(values)
        best = min(values,key=lambda x:(x[0],x[1]))
        if best[0] >= base[old]-1e-12: return old,evaluations
        selected = best[2]


def run():
    n, k = len(TICKERS), PORTFOLIO_K
    frame = market()
    close = close_panel()[TICKERS]
    op = frame.pivot(index="date",columns="ticker",values="open")[TICKERS]
    returns = close.pct_change(fill_method=None).dropna()
    rows, parameters = [], []
    for month in range(1,9):
        sessions = close.index[(close.index.year==2026)&(close.index.month==month)]
        start,end = sessions[0],sessions[-1]
        history = returns.loc[returns.index<start].tail(126)
        assert len(history)==126
        mu,cov = history.mean().to_numpy()*252,history.cov().to_numpy()*252
        bits,base,cost,h,j,penalty = encode(mu,cov)
        feasible = bits.sum(axis=1)==k
        tic=perf_counter()
        candidates=bits[feasible]
        exact_values=5*np.einsum("bi,ij,bj->b",candidates,cov,candidates)/k**2-candidates@mu/k
        exact=np.where(feasible)[0][np.argmin(exact_values)]
        exact_time=perf_counter()-tic
        tic=perf_counter(); local,n_eval=local_swap(base,bits,mu,k); local_time=perf_counter()-tic
        realised=close.loc[end].to_numpy()/op.loc[start].to_numpy()-1
        for name,idx,elapsed,evals in [("Exact",exact,exact_time,comb(n,k)),("Local swap",local,local_time,n_eval)]:
            rows.append(dict(month=month,start=start,end=end,method=name,depth=0,seed=0,objective=base[idx],
                gap=base[idx]-base[exact],feasible_probability=1.,optimal_probability=float(idx==exact),
                gross_return=float(bits[idx]@realised/k),net_return=float((1+bits[idx]@realised/k)*.998-1),
                seconds=elapsed,evaluations=evals,shots=0,holdings=",".join(np.array(TICKERS)[bits[idx].astype(bool)])))
        for depth in [1,2]:
            for seed in [11,29,47]:
                rng=np.random.default_rng(seed)
                tic=perf_counter()
                objective=lambda theta: float(qaoa_probabilities(theta,h,j)@cost)
                fit=minimize(objective,rng.uniform(.05,.5,2*depth),method="COBYLA",options={"maxiter":80,"rhobeg":.3})
                prob=qaoa_probabilities(fit.x,h,j)
                counts=rng.multinomial(2048,prob/prob.sum())
                found=np.where((counts>0)&feasible)[0]
                idx=int(found[np.argmin(base[found])]) if len(found) else None
                rows.append(dict(month=month,start=start,end=end,method="QAOA",depth=depth,seed=seed,
                    objective=float(base[idx]) if idx is not None else None,
                    gap=float(base[idx]-base[exact]) if idx is not None else None,
                    feasible_probability=float(prob[feasible].sum()),optimal_probability=float(prob[exact]),
                    gross_return=float(bits[idx]@realised/k) if idx is not None else None,
                    net_return=float((1+bits[idx]@realised/k)*.998-1) if idx is not None else None,
                    seconds=perf_counter()-tic,evaluations=fit.nfev,shots=2048,
                    holdings=",".join(np.array(TICKERS)[bits[idx].astype(bool)]) if idx is not None else "NONE"))
                parameters.append(dict(month=month,depth=depth,seed=seed,theta=fit.x,h=h,j=j,penalty=penalty,
                                       exact_index=int(exact),base=base,cost=cost))
        print(f"portfolio month {month} complete",flush=True)
    output=pd.DataFrame(rows)
    output.to_csv(RESULTS/"portfolio_runs.csv",index=False)
    save_json("portfolio_parameters.json",parameters)
    q=output[output.method=="QAOA"]
    summary={"n":n,"k":k,"feasible_states":comb(n,k),"dates":8,"runs":len(q),"shots":2048,
             "max_gap":q.gap.max(),"optimal_runs":int((q.gap.abs()<1e-10).sum()),
             "failure_runs":int(q.gap.isna().sum()),"mean_feasibility":q.feasible_probability.mean(),
             "mean_optimum_probability":q.optimal_probability.mean(),"mean_seconds":q.seconds.mean(),
             "by_depth":q.groupby("depth")[["gap","feasible_probability","optimal_probability","seconds"]].mean().to_dict(),
             "exact_monthly_returns":output[output.method=="Exact"].net_return.to_list(),
             "exact_compounded_net":float(np.prod(1+output[output.method=="Exact"].net_return)-1)}
    save_json("portfolio_summary.json",summary)
    sampling=[]
    for shots in [16,64,256,2048]:
        for depth in [1,2]:
            pstar=q.loc[q.depth==depth,"optimal_probability"].to_numpy()
            sampling.append(dict(shots=shots,method=f"QAOA p={depth}",success=float(np.mean(1-(1-pstar)**shots))))
        sampling.extend([dict(shots=shots,method="Uniform bit strings",success=1-(1-1/2**n)**shots),
                         dict(shots=shots,method="Uniform feasible",success=1-(1-1/comb(n,k))**shots)])
    pd.DataFrame(sampling).to_csv(RESULTS/"portfolio_sampling.csv",index=False)
    fig,ax=plt.subplots(1,2,figsize=(7,3.3))
    for depth,group in q.groupby("depth"):
        a=group.groupby("month").optimal_probability.agg(["mean","min","max"])
        ax[0].plot(a.index,a["mean"],marker="o",label=f"p={depth}")
        ax[0].fill_between(a.index,a["min"],a["max"],alpha=.14)
    ax[0].set(xlabel="2026 decision month",ylabel="Exact-optimum probability")
    ax[0].legend()
    for depth,group in q.groupby("depth"):
        ax[1].scatter(group.seconds,group.gap,s=24,label=f"p={depth}")
    ax[1].legend()
    ax[1].set(xlabel="Local optimisation + decoding (s)",ylabel="Selected objective gap")
    save_figure("portfolio_comparison")
    return summary


if __name__=="__main__": print(run())
