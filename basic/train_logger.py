"""
Training logs: one progress line per epoch on screen (replaced by the next epoch),
and one line per epoch in results/train_log.txt.
"""
import logging
import re
import shutil

from tqdm import tqdm

BAR_FORMAT = "{desc} {bar:10} {n_fmt}/{total_fmt} {elapsed}<{remaining} {metrics}"
# width of BAR_FORMAT without {metrics}, used to keep the line shorter than the terminal
BAR_OVERHEAD = 48
SCREEN_SECTIONS = ("train", "best", "clip")
FILE_SECTIONS = ("train", "lr", "clip", "val")
SHORT_NAMES = {"loss_ctc": "ctc", "loss_ce": "ce", "loss_ce_end": "ce_end"}


def format_metrics(values):
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


class ClipMonitor(logging.Filter):
    """
    Hides the gradient clipping warnings of ScaledAdam and keeps, per epoch, the mean
    clipped percentage and the number of steps scaled by less than 0.5.
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
        if self.windows:
            text = "clip %.1f%%" % (sum(self.windows) / len(self.windows))
        else:
            text = "clip n/a"
        if self.strong:
            text += " (%d strong)" % self.strong
        self.reset()
        return text


def model_summary(models):
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


class _MetricsBar(tqdm):
    # adds a {metrics} field to the bar format
    metrics = ""

    @property
    def format_dict(self):
        d = super(_MetricsBar, self).format_dict
        d["metrics"] = self.metrics
        return d


class EpochProgress:
    """
    Progress bar of one epoch. The bar is updated by hand instead of wrapping the loader,
    so that it stays open for the validation results after the last batch.
    """

    def __init__(self, iterable, epoch, nb_epochs, enabled=True, refresh_every=20,
                 leave=False, sticky=None):
        self.iterable = iterable
        self.desc = "Epoch %*d/%d" % (len(str(nb_epochs)), epoch, nb_epochs)
        self.refresh_every = refresh_every
        self.sections = dict(sticky or {})
        self.bar = _MetricsBar(total=len(iterable), desc=self.desc, bar_format=BAR_FORMAT,
                               leave=leave, dynamic_ncols=True, disable=not enabled)
        self._render()

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.bar.update(1)

    def live(self, ind_batch, display_values):
        if ind_batch % self.refresh_every == 0:
            self.set("train", format_metrics(display_values), refresh=False)

    def set(self, section, text, refresh=True):
        self.sections[section] = text
        self._render()
        if refresh:
            self.bar.refresh()

    def close(self):
        """Closes the bar and returns the line for train_log.txt."""
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
            joined = joined[:room - 3] + "..."
        self.bar.metrics = "| " + joined

    def _joined(self, order):
        return " | ".join(self.sections[k] for k in order if k in self.sections)


class TrainLog:

    def __init__(self, path):
        self.path = path

    def write(self, line):
        with open(self.path, "a") as f:
            f.write(line + "\n")
