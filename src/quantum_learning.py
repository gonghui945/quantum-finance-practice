"""Chronological weekly equity classification with classical and quantum kernels.

The eight-name panel is a teaching example, not a point-in-time market
universe. Hyperparameters use 2025 only; the final 2026 sample is never searched.
Cached statevectors accelerate simulation but are not available from a QPU.
"""
from __future__ import annotations
import argparse
import json
from time import perf_counter
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import spearmanr
import matplotlib.dates as mdates
from common import TICKERS, DATA, RESULTS, market, close_panel, save_json, plt

FEATURES=["week_return", "momentum4", "volatility20", "week_range"]


def weekly_panel():
    daily=market(); daily=daily[daily.ticker.isin(TICKERS)].copy()
    daily["week"]=daily.date-pd.to_timedelta(daily.date.dt.dayofweek,unit="D")
    days=pd.read_csv(DATA/"sessions.csv",parse_dates=["date"])
    days["week"]=days.date-pd.to_timedelta(days.date.dt.dayofweek,unit="D")
    expected=days.groupby("week").size()
    close=close_panel()[TICKERS]
    vol=close.pct_change(fill_method=None).rolling(20,min_periods=20).std()*np.sqrt(252)
    rows=[]
    for (ticker,week),g in daily.groupby(["ticker","week"]):
        # A week crossing the cutoff cannot supply a completed label.
        if week+pd.Timedelta(days=4)>pd.Timestamp("2026-08-31") or len(g)!=expected[week]: continue
        g=g.sort_values("date")
        rows.append(dict(ticker=ticker,week=week,end=g.date.iloc[-1],open=g.open.iloc[0],close=g.close.iloc[-1],
                         week_return=g.close.iloc[-1]/g.open.iloc[0]-1,
                         volatility20=vol.loc[g.date.iloc[-1],ticker],
                         week_range=(g.high.max()-g.low.min())/g.close.iloc[-1]))
    w=pd.DataFrame(rows)
    # Calendar joins, not row shifts: missing weeks must not turn into longer horizons.
    lag=w[["ticker","week","close"]].rename(columns={"close":"lag4"})
    lag["week"]+=pd.Timedelta(weeks=4)
    w=w.merge(lag,on=["ticker","week"],how="left")
    w["momentum4"]=w.close/w.lag4-1
    counts=w.groupby("week").ticker.nunique()
    target=w[w.week.isin(counts[counts==len(TICKERS)].index)][["ticker","week","end","week_return"]].copy()
    target["target"]=target.groupby("week").week_return.transform(lambda r:(r.rank(method="first")>len(TICKERS)//2).astype(int))
    target=target.rename(columns={"week":"label_week","end":"label_end","week_return":"forward_return"})
    w["label_week"]=w.week+pd.Timedelta(weeks=1)
    panel=w.merge(target,on=["ticker","label_week"],how="inner").dropna(subset=FEATURES)
    # Retain balanced, complete cohort cross-sections only.
    complete=panel.groupby("label_week").ticker.nunique()
    panel=panel[panel.label_week.isin(complete[complete==len(TICKERS)].index)].sort_values(["label_week","ticker"])
    assert (panel.end<panel.label_week).all()
    assert panel.label_end.max()<=pd.Timestamp("2026-08-31")
    panel.to_csv(DATA/"weekly_learning_panel.csv",index=False)
    return panel


def feature_state(x, layers=2, scale=1., entangle=True):
    n=len(x)
    dev=qml.device("default.qubit",wires=n)
    @qml.qnode(dev)
    def circuit():
        for layer in range(layers):
            for i in range(n):
                qml.RY(scale*x[i],wires=i)
                qml.RZ(scale*x[(i+1)%n],wires=i)
            if entangle:
                for i in range(n-1): qml.CNOT(wires=[i,i+1])
        return qml.state()
    return np.asarray(circuit())


def states(x,layers=2,scale=1.,entangle=True):
    return np.array([feature_state(row,layers,scale,entangle) for row in x])


def fidelity(a,b): return np.abs(a.conj()@b.T)**2


def product_kernel(a,b,scale=1.):
    # H followed by Rz per qubit yields this exactly classically evaluable kernel.
    return np.prod(np.cos(scale*(a[:,None,:]-b[None,:,:])/2)**2,axis=-1)


def two_axis_kernel(a,b,scale=1.):
    """Classical closed form of the one-layer Ry/Rz feature-map fidelity."""
    u,v=scale*a[:,None,:],scale*b[None,:,:]
    overlap=(1+np.cos(u)*np.cos(v)+np.sin(u)*np.sin(v)*np.cos(np.roll(u,-1,axis=-1)-np.roll(v,-1,axis=-1)))/2
    return np.prod(overlap,axis=-1)


def product_state(x,entangle=True):
    n=len(x); dev=qml.device("default.qubit",wires=n)
    @qml.qnode(dev)
    def circuit():
        for i in range(n): qml.Hadamard(wires=i); qml.RZ(x[i],wires=i)
        if entangle:
            for i in range(n-1): qml.CNOT(wires=[i,i+1])
        return qml.state()
    return np.asarray(circuit())


def scores(y,s):
    return {"auc":float(roc_auc_score(y,s)),"balanced_accuracy":float(balanced_accuracy_score(y,np.asarray(s)>0))}


def train_vqc(x,y,test,seed=11,steps=100):
    """Small fixed-budget variational classifier; exact simulator backpropagation."""
    pnp=qml.numpy; rng=np.random.default_rng(seed); n=x.shape[1]
    dev=qml.device("default.qubit",wires=n)
    @qml.qnode(dev,interface="autograd",diff_method="backprop")
    def circuit(weights,features):
        for layer in range(2):
            for i in range(n):
                qml.RY(features[:,i],wires=i)
                qml.RZ(weights[layer,i,0],wires=i)
                qml.RY(weights[layer,i,1],wires=i)
            for i in range(n-1): qml.CNOT(wires=[i,i+1])
        return qml.expval(qml.PauliZ(0))
    weights=pnp.array(rng.normal(0,.1,(2,n,2)),requires_grad=True)
    optimizer=qml.AdamOptimizer(.05)
    losses=[]
    for step in range(steps):
        idx=rng.choice(len(x),min(32,len(x)),replace=False)
        xx=pnp.array(x[idx],requires_grad=False); yy=pnp.array(y[idx],requires_grad=False)
        def loss(w):
            p=(1+circuit(w,xx))*.5
            p=1e-6+(1-2e-6)*p
            return -pnp.mean(yy*pnp.log(p)+(1-yy)*pnp.log(1-p))
        weights,value=optimizer.step_and_cost(loss,weights)
        losses.append(float(value))
    pred=np.asarray(circuit(weights,pnp.array(test,requires_grad=False)))
    return pred,losses


def plot_results(metrics, weekly, selected):
    """Render Figure 3.1 without fitting models or modifying saved statistics."""
    chosen_layers = int(next(c["layers"] for c in selected
                             if c["method"] == "Reuploading kernel"))
    short = metrics[~metrics.method.str.startswith("VQC")]
    labels = short.method.replace({
        "Reuploading kernel": f"Rotation kernel\n(selected $L={chosen_layers}$)",
        "Depth-two diagnostic": "Rotation kernel\n($L=2$ check)",
        "Product kernel": "Product\nkernel",
        "Logistic": "Logistic\nregression",
    })
    blue, orange = "#174a68", "#ba6035"
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.55),
                           gridspec_kw={"width_ratios": [1, 1.28]})
    # Wrapped model labels reclaim space; the legend stays outside the data.
    fig.subplots_adjust(left=.16, right=.985, bottom=.18, top=.80, wspace=.48)
    colours = [orange if name == "Reuploading kernel" else blue
               for name in short.method]
    bars = ax[0].barh(labels, short.auc, height=.63, color=colours, zorder=3)
    # Redundant marks keep the selected model identifiable in grayscale.
    for bar, name in zip(bars, short.method):
        if name == "Reuploading kernel":
            bar.set_hatch("///")
            bar.set_edgecolor("#222222")
            bar.set_linewidth(.7)
    ax[0].axvline(.5, color="#555555", ls="--", lw=.8, zorder=4)
    for bar, value in zip(bars, short.auc):
        ax[0].text(max(value + .025, .535), bar.get_y() + bar.get_height() / 2,
                   f"{value:.4f}", va="center", fontsize=8.5, color="#333333")
    ax[0].set(xlim=(0, 1), xlabel="2026 held-out AUC")
    ax[0].set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.50", "0.75", "1"])
    ax[0].tick_params(axis="y", length=0, pad=7, labelsize=9)
    ax[0].set_axisbelow(True)
    ax[0].grid(axis="x", color="#e5e5e5", linewidth=.6)

    ic = weekly.pivot(index="week", columns="method", values="ic")
    ic.index = pd.to_datetime(ic.index)
    calendar = pd.date_range("2026-01-05", "2026-08-24", freq="W-MON")
    for name, colour, label, linestyle, marker in [
        ("RBF SVC", blue, "RBF SVC", "--", "s"),
        ("Reuploading kernel", orange, f"Rotation ($L={chosen_layers}$)", "-", "o"),
    ]:
        # Average four retained weeks, then reinsert excluded calendar weeks.
        # Do not interpolate across gaps or silently change the estimand.
        plotted = ic[name].rolling(4).mean().reindex(calendar)
        ax[1].plot(plotted.index, plotted, label=label, color=colour, lw=1.5,
                   linestyle=linestyle, marker=marker, markersize=3.2,
                   markerfacecolor="white", markeredgewidth=.8)
    ax[1].axhline(0, color="#555555", lw=.8)
    ax[1].legend(loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=2,
                 frameon=False, fontsize=8, handlelength=2.3,
                 columnspacing=1.1, borderaxespad=0)
    ax[1].set(xlabel="2026 label week", ylabel="Mean rank IC\n(four retained weeks)")
    ax[1].yaxis.label.set_size(9)
    ax[1].set_xlim(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-08-31"))
    ax[1].xaxis.set_major_locator(mdates.MonthLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[1].grid(axis="y", color="#e5e5e5", linewidth=.6)
    ax[1].set_axisbelow(True)
    ax[1].tick_params(axis="x", labelsize=8.5, rotation=0)
    for axis, title in zip(ax, ["(a) Classification", "(b) Weekly ranking"]):
        axis.set_title(title, loc="left", y=1.13, fontsize=10)
    fig.savefig(RESULTS / "qml_comparison.pdf", bbox_inches="tight", pad_inches=.06)
    fig.savefig(RESULTS / "qml_comparison.png", dpi=200,
                bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    return fig


def run():
    panel=weekly_panel()
    train=panel[panel.label_week.dt.year==2024]
    validation=panel[panel.label_week.dt.year==2025]
    test=panel[panel.label_week.dt.year==2026]
    scaler=StandardScaler().fit(train[FEATURES])
    transform=lambda f: np.clip(scaler.transform(f[FEATURES]),-3,3)*np.pi/3
    x,v=transform(train),transform(validation)
    y,yv=train.target.to_numpy(),validation.target.to_numpy()
    searches=[]
    for c in [.1,1.,10.]:
        clf=LogisticRegression(C=c,max_iter=2000).fit(x,y)
        searches.append(dict(method="Logistic",C=c,scale=0.,gamma=0.,layers=0,auc=roc_auc_score(yv,clf.decision_function(v))))
        for gamma in [.1,1.]:
            clf=SVC(C=c,gamma=gamma).fit(x,y)
            searches.append(dict(method="RBF SVC",C=c,scale=0.,gamma=gamma,layers=0,auc=roc_auc_score(yv,clf.decision_function(v))))
    for method in ["Product kernel","Reuploading kernel"]:
        for scale in [.5,1.]:
            for layers in ([1] if method=="Product kernel" else [1,2]):
                if method=="Product kernel": kt,kv=product_kernel(x,x,scale),product_kernel(v,x,scale)
                else:
                    sx,sv=states(x,layers,scale),states(v,layers,scale)
                    kt,kv=fidelity(sx,sx),fidelity(sv,sx)
                for c in [.1,1.,10.]:
                    clf=SVC(C=c,kernel="precomputed").fit(kt,y)
                    searches.append(dict(method=method,C=c,scale=scale,gamma=0.,layers=layers,auc=roc_auc_score(yv,clf.decision_function(kv))))
    search=pd.DataFrame(searches)
    search.to_csv(RESULTS/"qml_validation_search.csv",index=False)
    selected=search.sort_values("auc",ascending=False,kind="stable").groupby("method",sort=False).head(1)
    # Refit on all eligible pre-2026 rows, with a deterministic latest-52-week cap.
    hist=panel[panel.label_week.dt.year<2026]
    weeks=sorted(hist.label_week.unique())[-52:]
    fit=hist[hist.label_week.isin(weeks)]
    scaler=StandardScaler().fit(fit[FEATURES])
    xf,xt=transform(fit),transform(test); yf=fit.target.to_numpy()
    predictions=test[["ticker","week","end","label_week","label_end","target","forward_return"]].copy()
    diagnostics=[]
    for config in selected.to_dict("records"):
        name=config["method"]; tic=perf_counter(); extra={}
        if name=="Logistic": model=LogisticRegression(C=config["C"],max_iter=2000).fit(xf,yf); pred=model.decision_function(xt)
        elif name=="RBF SVC": model=SVC(C=config["C"],gamma=config["gamma"]).fit(xf,yf); pred=model.decision_function(xt)
        else:
            if name=="Product kernel": kf,kx=product_kernel(xf,xf,config["scale"]),product_kernel(xt,xf,config["scale"])
            else:
                sf,st=states(xf,int(config["layers"]),config["scale"]),states(xt,int(config["layers"]),config["scale"])
                kf,kx=fidelity(sf,sf),fidelity(st,sf)
                np.savez_compressed(RESULTS/"qml_kernel_inputs.npz",xf=xf,xt=xt,yf=yf,yt=test.target.to_numpy(),kf=kf,kx=kx,
                                    layers=int(config["layers"]),scale=config["scale"])
            eigen=np.linalg.eigvalsh(kf)
            extra={"effective_rank":float(np.trace(kf)**2/np.sum(kf*kf)),
                   "min_eigenvalue":float(eigen.min()),"offdiag_std":float(kf[np.triu_indices(len(kf),1)].std())}
            model=SVC(C=config["C"],kernel="precomputed").fit(kf,yf); pred=model.decision_function(kx)
        predictions[name]=pred
        diagnostics.append(dict(method=name,seconds=perf_counter()-tic,**scores(test.target,pred),**extra))
    # Report the depth-two branch selected on validation, without replacing the
    # validation winner after looking at test data.
    deep=search[(search.method=="Reuploading kernel")&(search.layers==2)].sort_values("auc",ascending=False).iloc[0]
    tic=perf_counter();sf,st=states(xf,2,deep.scale),states(xt,2,deep.scale)
    model=SVC(C=deep.C,kernel="precomputed").fit(fidelity(sf,sf),yf)
    pred=model.decision_function(fidelity(st,sf));name="Depth-two diagnostic"
    predictions[name]=pred
    diagnostics.append(dict(method=name,seconds=perf_counter()-tic,**scores(test.target,pred)))
    for seed in [11,29,47]:
        tic=perf_counter(); pred,loss=train_vqc(xf,yf,xt,seed)
        name=f"VQC {seed}"; predictions[name]=pred
        diagnostics.append(dict(method=name,seconds=perf_counter()-tic,**scores(test.target,pred)))
        save_json(f"qml_vqc_loss_{seed}.json",loss)
    models=[d["method"] for d in diagnostics]
    weekly=[]
    for week,g in predictions.groupby("label_week"):
        for name in models:
            top=g.nlargest(2,name)
            weekly.append(dict(week=week,method=name,ic=spearmanr(g[name],g.forward_return).statistic,
                               top2_return=top.forward_return.mean(),cohort_return=g.forward_return.mean(),
                               top2_net=(1+top.forward_return.mean())*.998-1))
    weekly=pd.DataFrame(weekly)
    checks_x=xf[:16]
    product_states=np.array([product_state(row) for row in checks_x])
    product_error=float(np.max(np.abs(fidelity(product_states,product_states)-product_kernel(checks_x,checks_x))))
    assert product_error<1e-12
    rotation_states=states(checks_x,layers=1)
    rotation_error=float(np.max(np.abs(fidelity(rotation_states,rotation_states)-two_axis_kernel(checks_x,checks_x))))
    assert rotation_error<1e-12
    predictions.to_csv(RESULTS/"qml_predictions.csv",index=False)
    weekly.to_csv(RESULTS/"qml_weekly.csv",index=False)
    diag=pd.DataFrame(diagnostics).merge(weekly.groupby("method").agg(mean_ic=("ic","mean"),mean_top2=("top2_return","mean"),mean_cohort=("cohort_return","mean")),on="method")
    diag.to_csv(RESULTS/"qml_metrics.csv",index=False)
    # The same whole-week resamples assess AUC and IC on frozen predictions.
    from qml_uncertainty import run as run_uncertainty
    uncertainty = run_uncertainty()
    primary = uncertainty["primary"]
    summary={"training_2024":len(train),"validation_2025":len(validation),"refit":len(fit),"test":len(test),
             "test_weeks":test.label_week.nunique(),"test_first":test.label_week.min(),"test_last_label_end":test.label_end.max(),
             "selected":selected.to_dict("records"),"metrics":diag.to_dict("records"),
             "depth_two_config":deep.to_dict(),
             "test_excluded_weeks":[str(w.date()) for w in pd.date_range("2026-01-05","2026-08-24",freq="W-MON") if w not in set(test.label_week)],
             "mean_paired_ic_difference":primary["delta_ic"],
             "conditional_block95":[primary["ic_low"], primary["ic_high"]],
             "uncertainty_method":uncertainty["estimator"],
             "product_identity_max_error":product_error,"features":FEATURES,
             "two_axis_identity_max_error":rotation_error,
             "simulator":"PennyLane default.qubit; exact cached statevectors; no hardware run"}
    save_json("qml_summary.json",summary)
    plot_results(diag, weekly, selected.to_dict("records"))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true",
                        help="Redraw Figure 3.1 from saved results without refitting models.")
    args = parser.parse_args()
    if args.plot_only:
        summary = json.loads((RESULTS / "qml_summary.json").read_text())
        plot_results(pd.read_csv(RESULTS / "qml_metrics.csv"),
                     pd.read_csv(RESULTS / "qml_weekly.csv"), summary["selected"])
    else:
        print(run())
