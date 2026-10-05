"""Market-calibrated model pricing and empirical tail-probability circuits.

Historical volatility is only an illustrative model calibration, not an implied
volatility quote. Future payoffs are synthetic draws under stated assumptions.
"""
from itertools import product
import numpy as np
import pandas as pd
import pennylane as qml
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from common import RESULTS, close_panel, save_json, save_figure, plt


def amplitude_preparation(prob,values):
    """Unitary A: load probabilities then encode a bounded payoff in an ancilla."""
    n=int(np.log2(len(prob)))
    qml.MottonenStatePreparation(np.sqrt(prob),wires=list(range(n)))
    for index,bits in enumerate(product([0,1],repeat=n)):
        qml.ctrl(qml.RY,control=list(range(n)),control_values=list(bits))(
            2*np.arcsin(np.sqrt(values[index])),wires=n)


def amplitude_matrix(prob,values):
    n=int(np.log2(len(prob)))
    return np.asarray(qml.matrix(amplitude_preparation,wire_order=list(range(n+1)))(prob,values))


def run():
    returns=close_panel().AAPL.pct_change(fill_method=None).dropna().tail(252)
    sigma=float(returns.std()*np.sqrt(252)); s0=100.;strike=100.;rate=.03;tenor=.25
    d1=(np.log(s0/strike)+(rate+sigma*sigma/2)*tenor)/(sigma*np.sqrt(tenor))
    bs=s0*norm.cdf(d1)-strike*np.exp(-rate*tenor)*norm.cdf(d1-sigma*np.sqrt(tenor))
    rows=[]
    for bins in [8,16,32,64,128]:
        edges=np.linspace(-4,4,bins+1);z=(edges[:-1]+edges[1:])/2
        raw=np.diff(norm.cdf(edges));prob=raw/raw.sum()
        terminal=s0*np.exp((rate-.5*sigma**2)*tenor+sigma*np.sqrt(tenor)*z)
        payoff=np.maximum(terminal-strike,0);cap=payoff.max();values=payoff/cap
        expected=float(prob@values)
        rows.append(dict(bins=bins,price=float(np.exp(-rate*tenor)*prob@payoff),
                         black_scholes=float(bs),absolute_error=float(abs(np.exp(-rate*tenor)*prob@payoff-bs))))
        if bins==16:
            A=amplitude_matrix(prob,values);state=A[:,0]
            circuit_amp=float(np.abs(state[1::2])@np.abs(state[1::2]))
            assert abs(circuit_amp-expected)<1e-12
            dim=len(state);s_zero=np.eye(dim);s_zero[0,0]=-1
            s_good=np.diag([1 if i%2==0 else -1 for i in range(dim)])
            Q=-A@s_zero@A.conj().T@s_good
            powers=np.array([0,1,2,4]);amplitudes=[]
            for m in powers:
                amplified=np.linalg.matrix_power(Q,int(m))@state
                amplitudes.append(float(np.sum(abs(amplified[1::2])**2)))
            theta=np.arcsin(np.sqrt(expected))
            assert np.max(np.abs(amplitudes-np.sin((2*powers+1)*theta)**2))<1e-11
            selected=dict(prob=prob,values=values,cap=cap,expected=expected,powers=powers,
                          amplitudes=amplitudes,price=float(np.exp(-rate*tenor)*cap*expected))
    grid=pd.DataFrame(rows);grid.to_csv(RESULTS/"pricing_discretisation.csv",index=False)
    rng=np.random.default_rng(2026); powers=selected["powers"];shots=250
    calls=int(np.sum(2*powers+1)*shots)
    theta_grid=np.linspace(1e-7,np.pi/2-1e-7,20001)
    probs=np.clip(np.sin(np.outer(theta_grid,2*powers+1))**2,1e-12,1-1e-12)
    repeats=[]
    for repeat in range(100):
        count=rng.binomial(shots,selected["amplitudes"])
        ll=(np.log(probs)*count+np.log1p(-probs)*(shots-count)).sum(axis=1)
        best=int(np.argmax(ll));lo=theta_grid[max(best-1,0)];hi=theta_grid[min(best+1,len(theta_grid)-1)]
        def loss(t):
            p=np.clip(np.sin((2*powers+1)*t)**2,1e-12,1-1e-12)
            return -float(np.sum(count*np.log(p)+(shots-count)*np.log1p(-p)))
        fit=minimize_scalar(loss,bounds=(lo,hi),method="bounded")
        ae=np.sin(fit.x)**2
        direct=rng.binomial(calls,selected["expected"])/calls
        multiplier=np.exp(-rate*tenor)*selected["cap"]
        repeats.append(dict(repeat=repeat,mlae=ae*multiplier,direct=direct*multiplier,
                            target=selected["price"],preparation_calls=calls))
    repeated=pd.DataFrame(repeats);repeated.to_csv(RESULTS/"pricing_repeated.csv",index=False)
    # An arithmetic Asian extension uses synthetic paths, never post-cutoff prices.
    z=rng.normal(size=(100000,32));dt=tenor/32
    paths=s0*np.exp(np.cumsum((rate-.5*sigma*sigma)*dt+sigma*np.sqrt(dt)*z,axis=1))
    asian=np.exp(-rate*tenor)*np.maximum(paths.mean(axis=1)-strike,0)
    # Tail risk: discrete expected shortfall includes just the necessary VaR atom.
    loss=-returns.to_numpy();alpha=.95;var=float(np.quantile(loss,alpha,method="inverted_cdf"))
    historical_es=float(var+np.mean(np.maximum(loss-var,0))/(1-alpha))
    edges=np.linspace(loss.min()-1e-9,loss.max()+1e-9,16+1)
    counts,_=np.histogram(loss,edges);p=counts/counts.sum();centres=(edges[:-1]+edges[1:])/2
    j=int(np.searchsorted(np.cumsum(p),alpha));grid_var=float(centres[j])
    grid_es=float(grid_var+p@np.maximum(centres-grid_var,0)/(1-alpha))
    indicator=(centres>grid_var).astype(float)
    tail_matrix=amplitude_matrix(p,indicator);tail=float(np.sum(abs(tail_matrix[1::2,0])**2))
    assert abs(tail-p@indicator)<1e-12
    payoff=np.maximum(centres-grid_var,0);cap=payoff.max()
    moment_matrix=amplitude_matrix(p,payoff/cap)
    moment=float(np.sum(abs(moment_matrix[1::2,0])**2))
    shots_risk=10000
    tail_sample=rng.binomial(shots_risk,tail)/shots_risk
    excess_sample=cap*rng.binomial(shots_risk,moment)/shots_risk
    sample_es=grid_var+excess_sample/(1-alpha)
    risk=dict(observations=len(loss),alpha=alpha,historical_var=var,historical_es=historical_es,
              grid_var=grid_var,grid_es=grid_es,strict_tail=tail,sampled_tail=tail_sample,
              sampled_es=sample_es,shots_per_circuit=shots_risk,
              es_measurement_se=cap*np.sqrt(moment*(1-moment)/shots_risk)/(1-alpha))
    pd.DataFrame({"loss_bin":centres,"probability":p,"excess":payoff}).to_csv(RESULTS/"risk_grid.csv",index=False)
    summary=dict(calibration_end=str(returns.index.max().date()),calibration_n=len(returns),sigma=sigma,
                 s0=s0,strike=strike,rate=rate,tenor=tenor,black_scholes=bs,
                 discretisation=rows,mlae_bins=16,mlae_target=selected["price"],preparation_calls=calls,
                 powers=powers,shots_per_power=shots,repeats=100,
                 mlae_rmse=float(np.sqrt(np.mean((repeated.mlae-repeated.target)**2))),
                 direct_rmse=float(np.sqrt(np.mean((repeated.direct-repeated.target)**2))),
                 asian_mc=float(asian.mean()),asian_mc_se=float(asian.std(ddof=1)/np.sqrt(len(asian))),
                 risk=risk)
    save_json("pricing_risk_summary.json",summary)
    fig,ax=plt.subplots(1,2,figsize=(7,3.3))
    ax[0].plot(grid.bins,grid.absolute_error,marker="o");ax[0].set_xscale("log",base=2)
    ax[0].set(xlabel="Gaussian grid bins",ylabel="Absolute call-price error (USD)")
    ax[1].boxplot([repeated.direct-repeated.target,repeated.mlae-repeated.target],tick_labels=["Direct","MLAE"])
    ax[1].axhline(0,color="black",lw=.8);ax[1].set(ylabel="Price error versus 16-bin target (USD)")
    save_figure("pricing_errors")
    return summary


if __name__=="__main__": print(run())
