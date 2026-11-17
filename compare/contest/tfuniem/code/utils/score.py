"""Local scoring. Proxy score is the leader's formula (NOT the official score):
  parameter_proxy = 1 / (1 + mean_j(RMSE_j / std_j)),  std_j = training-side std of valid labels (ddof=0)
  proxy = 30 * protocol_acc + 30 * modulation_acc + 40 * parameter_proxy
"""
import numpy as np

from utils.common import NARROWBAND_MODS, TARGETS


def classification_metrics(pred, true, names):
    n = len(names)
    cm = np.zeros((n, n), np.int64)
    np.add.at(cm, (true, pred), 1)
    support = cm.sum(1)
    recall = np.where(support > 0, np.diag(cm) / np.maximum(support, 1), np.nan)
    return {'n': int(len(true)), 'accuracy': float((pred == true).mean()) if len(true) else float('nan'),
            'macro_recall': float(np.nanmean(recall)),
            'per_class_recall': {names[i]: (None if np.isnan(recall[i]) else round(float(recall[i]), 4))
                                 for i in range(n)},
            'confusion': cm.tolist()}


def parameter_metrics(pred, truth, valid, env_snr=None, env_bins=None):
    """pred/truth: (N, 7) raw units; valid: (N, 7) bool."""
    out = {}
    for j, name in enumerate(TARGETS):
        v = valid[:, j]
        e = pred[v, j] - truth[v, j]
        out[name] = {'count': int(v.sum()), 'rmse': float(np.sqrt(np.mean(e ** 2))) if v.any() else None,
                     'mae': float(np.mean(np.abs(e))) if v.any() else None,
                     'abs_p95': float(np.quantile(np.abs(e), .95)) if v.any() else None}
        if env_snr is not None and j in (3, 4, 5) and v.any():
            edges = [-np.inf] + list(env_bins) + [np.inf]
            seg = {}
            for a, b in zip(edges[:-1], edges[1:]):
                s = v & (env_snr > a) & (env_snr <= b)
                if s.any():
                    es = pred[s, j] - truth[s, j]
                    seg[f'({a:.2f},{b:.2f}]'] = {'count': int(s.sum()), 'rmse': float(np.sqrt(np.mean(es ** 2))),
                                                  'mae': float(np.mean(np.abs(es)))}
            out[name]['by_env_snr_db'] = seg
    return out


def proxy(protocol_acc, modulation_acc, params, std):
    ratios = [params[name]['rmse'] / std[j] for j, name in enumerate(TARGETS) if params[name]['rmse'] is not None]
    pp = 1.0 / (1.0 + float(np.mean(ratios)))
    return pp, 30 * protocol_acc + 30 * modulation_acc + 40 * pp


def score_all(task, mod_pred, mod_true, pro_pred, pro_true, par_pred, par_true, par_valid, schema, std,
              env_snr=None, env_bins=None):
    """All arrays aligned on rows. task: 0 modulation, 1 protocol, 2 parameter."""
    mod_names = schema['classification_labels']['modulation']
    pro_names = schema['classification_labels']['protocol']
    m, p, r = task == 0, task == 1, task == 2
    mod = classification_metrics(mod_pred[m], mod_true[m], mod_names)
    nb = np.isin(mod_true[m], [mod_names.index(c) for c in NARROWBAND_MODS])
    mod['narrowband14_accuracy'] = float((mod_pred[m][nb] == mod_true[m][nb]).mean())
    mod['other9_accuracy'] = float((mod_pred[m][~nb] == mod_true[m][~nb]).mean())
    pro = classification_metrics(pro_pred[p], pro_true[p], pro_names)
    par = parameter_metrics(par_pred[r], par_true[r], par_valid[r],
                            None if env_snr is None else env_snr[r], env_bins)
    pp, score = proxy(pro['accuracy'], mod['accuracy'], par, std)
    return {'proxy_score': score, 'parameter_proxy': pp, 'protocol_accuracy': pro['accuracy'],
            'modulation_accuracy': mod['accuracy'], 'modulation': mod, 'protocol': pro, 'parameters': par,
            'note': 'Local proxy (leader formula, std_j = training-side std). NOT the official score.'}
