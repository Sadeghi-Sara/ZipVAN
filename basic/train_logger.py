"""
Concise training logs for VAN.

Screen: ONE live line that is replaced every epoch (nothing piles up):

  Epoch  85/500 ━━━━━━━━━━━━ 288/1621 00:09<00:42 31.5it/s | ctc 2.979 cer 0.7905 wer 0.9789 | best val cer 0.0341 wer 0.1210 @84 | clip 0.0%

File (results/train_log.txt): one permanent line per epoch, with everything:

  Epoch  85/500 |   52s | ctc 2.979 cer 0.7905 wer 0.9789 | lr 2.92e-02 | clip 0.0% | val cer 0.0341 wer 0.1210 *

- a model summary table when training starts
- ScaledAdam's clipping warnings are swallowed and reported as one per-epoch
  percentage (plus a count of strong clips, where gradients were scaled < 0.5)
"""
import logging
import re
import shutil

from tqdm import tqdm

BAR_FORMAT = "{desc} {bar:10} {n_fmt}/{total_fmt} {elapsed}<{remaining} {metrics}"
# width of everything in BAR_FORMAT except {metrics}; used to truncate the metrics
# so the line never exceeds the terminal and wraps (a wrapped tqdm line garbles
# itself, because \r only returns to the start of the last visual row).
BAR_OVERHEAD = 48
SCREEN_SECTIONS = ("train", "best", "clip")            # kept short so the line is not cut off
FILE_SECTIONS = ("train", "lr", "clip", "val")
SHORT_NAMES = {"loss_ctc": "ctc", "loss_ce": "ce", "loss_ce_end": "ce_end"}


# --------------------------------------------------------------------------- #
def format_metrics(values):
    """{'loss_ctc': 25.01, 'cer': 0.7512, ...} -> 'ctc 25.010 cer 0.7512 ...'
    CER/WER are shown as fractions, like in the predict_*.txt files (WER can exceed 1)."""
    parts = []
    for key, value in values.items():
        name = SHORT_NAMES.get(key, key)
        if key in ("cer", "wer", "cer_force_len"):
            parts.append("%s %.4f" % (name, value))
        elif key.startswith("loss"):
            parts.append("%s %.3f" % (name, value))
        elif key == "diff_len":
            parts.append("%s %.2f" % (name, value))
        else:
            parts.append("%s %.4g" % (name, value))
    return " ".join(parts)


# --------------------------------------------------------------------------- #
class ClipMonitor(logging.Filter):
    """Swallows ScaledAdam's gradient-clipping warnings and keeps per-epoch stats.

    ScaledAdam logs 'percent-clipped=X' once every clipping_update_period (100)
    steps, and 'Scaling gradients by ...' whenever a step is scaled below 0.5.
    """
    _PCT = re.compile(r"percent-clipped=([0-9.]+)")
    _SWALLOW = ("Clipping_scale=", "Scaling gradients by",
                "Parameter dominating tot_sumsq", "Parameters with most larger-than-usual grads")

    def __init__(self):
        super(ClipMonitor, self).__init__()
        self.reset()

    @classmethod
    def install(cls):
        root = logging.getLogger()
        for f in list(root.filters):
            if isinstance(f, cls):
                root.removeFilter(f)
        monitor = cls()
        root.addFilter(monitor)
        return monitor

    def reset(self):
        self.windows = []
        self.strong = 0

    def filter(self, record):
        msg = record.getMessage()
        if not msg.startswith(self._SWALLOW):
            return True
        if msg.startswith("Clipping_scale="):
            m = self._PCT.search(msg)
            if m:
                self.windows.append(float(m.group(1)))
        elif msg.startswith("Scaling gradients by"):
            self.strong += 1
        return False

    def epoch_text(self):
        """Summary for the epoch that just ended, then reset for the next one."""
        if self.windows:
            text = "clip %.1f%%" % (sum(self.windows) / len(self.windows))
        else:
            text = "clip n/a"
        if self.strong:
            text += " (%d strong)" % self.strong
        self.reset()
        return text


# --------------------------------------------------------------------------- #
def model_summary(models):
    """Table of models with parameter counts, as a printable string."""
    rows, total, trainable = [], 0, 0
    for name, model in models.items():
        n = sum(p.numel() for p in model.parameters())
        t = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows.append((name, type(model).__name__, n, t))
        total += n
        trainable += t
    w_name = max([len("name")] + [len(r[0]) for r in rows])
    w_cls = max([len("type")] + [len(r[1]) for r in rows])
    line = "  " + "-" * (w_name + w_cls + 30)
    out = [line, "  %-*s  %-*s %13s %13s" % (w_name, "name", w_cls, "type", "params", "trainable"), line]
    for name, cls, n, t in rows:
        out.append("  %-*s  %-*s %13s %13s" % (w_name, name, w_cls, cls, "{:,}".format(n), "{:,}".format(t)))
    out += [line,
            "  Total params         : {:,}".format(total),
            "  Trainable params     : {:,}".format(trainable),
            "  Non-trainable params : {:,}".format(total - trainable),
            "  Params size (fp32)   : %.2f MB" % (total * 4 / 1e6),
            ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
class _MetricsBar(tqdm):
    """tqdm with a `{metrics}` field. tqdm's own `{postfix}` always gets a ', '
    prefix; extra format_dict keys are passed to the format string untouched."""
    metrics = ""

    @property
    def format_dict(self):
        d = super(_MetricsBar, self).format_dict
        d["metrics"] = self.metrics
        return d


class EpochProgress:
    """Live bar for one epoch.

    leave=False: the line is erased when the epoch ends, so the next epoch's bar
    takes its place. The full record goes to the log file via close().

    The bar is advanced manually instead of wrapping the loader: tqdm closes a
    wrapped iterable as soon as the loop ends, which would freeze the line before
    clip / validation results could be added to it.
    """

    def __init__(self, iterable, epoch, nb_epochs, enabled=True, refresh_every=20,
                 leave=False, sticky=None):
        self.iterable = iterable
        self.desc = "Epoch %*d/%d" % (len(str(nb_epochs)), epoch, nb_epochs)
        self.refresh_every = refresh_every
        self.sections = dict(sticky or {})       # e.g. {"best": "best val cer 3.41% @84"}
        self.bar = _MetricsBar(total=len(iterable), desc=self.desc, bar_format=BAR_FORMAT,
                               ascii=" ━", leave=leave, dynamic_ncols=True, disable=not enabled)
        self._render()

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.bar.update(1)

    def live(self, ind_batch, display_values):
        """Update the running train metrics every `refresh_every` batches."""
        if ind_batch % self.refresh_every == 0:
            self.set("train", format_metrics(display_values), refresh=False)

    def set(self, section, text, refresh=True):
        self.sections[section] = text
        self._render()
        if refresh:
            self.bar.refresh()

    def close(self):
        """Close the bar and return the full plain line for the log file."""
        elapsed = self.bar.format_dict.get("elapsed", 0.0)
        self.bar.close()
        return "%s | %4.0fs | %s" % (self.desc, elapsed, self._joined(FILE_SECTIONS))

    def _render(self):
        joined = self._joined(SCREEN_SECTIONS)
        if not joined:
            self.bar.metrics = ""
            return
        room = max(20, shutil.get_terminal_size((110, 24)).columns - BAR_OVERHEAD)
        if len(joined) > room:
            joined = joined[:room - 1] + "…"
        self.bar.metrics = "| " + joined

    def _joined(self, order):
        return " | ".join(self.sections[k] for k in order if k in self.sections)


class TrainLog:
    """Appends one plain line per epoch to a text file (survives restarts)."""

    def __init__(self, path):
        self.path = path

    def write(self, line):
        with open(self.path, "a") as f:
            f.write(line + "\n")
