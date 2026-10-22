#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is 
#  to automatically recognize text from text-block images following a single-column layout.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

from basic.generic_training_manager import GenericTrainingManager
from basic.utils import edit_wer_from_list, nb_words_from_list, nb_chars_from_list, LM_ind_to_str
import editdistance
import re
import torch
from torch.nn import CTCLoss
from torch.amp import autocast


class TrainerLineCTC(GenericTrainingManager):

    def __init__(self, params):
        super(TrainerLineCTC, self).__init__(params)

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def ctc_loss(self):
        return CTCLoss(blank=self.dataset.tokens["blank"],
                       reduction=self.params["training_params"].get("ctc_reduction", "sum"),
                       zero_infinity=self.params["training_params"].get("ctc_zero_infinity", False))

    def passes_per_sample(self):
        return 2 if self.params["training_params"].get("cr_alpha", 0) > 0 else 1

    def cr_loss(self, log_pa, log_pb, x_reduced_len, y_len):
        """
        Consistency term of CR-CTC: 0.5 * (KL(sg(pb) || pa) + KL(sg(pa) || pb)) summed over
        the valid frames of each line, divided by the number of characters of the line and
        averaged over the batch (the normalization of CTCLoss with reduction="mean").
        log_pa, log_pb: (B, vocab_size + 1, W) log-probabilities of the two views
        """
        B, _, W = log_pa.shape
        mask = torch.zeros(B, W, device=log_pa.device)
        for i, l in enumerate(x_reduced_len):
            mask[i, :l] = 1.0
        pa, pb = log_pa.exp(), log_pb.exp()
        kl_ab = (pb.detach() * (log_pb.detach() - log_pa)).sum(dim=1)
        kl_ba = (pa.detach() * (log_pa.detach() - log_pb)).sum(dim=1)
        per_sample = (0.5 * (kl_ab + kl_ba) * mask).sum(dim=1)
        target_len = torch.as_tensor(y_len, dtype=per_sample.dtype, device=per_sample.device)
        return (per_sample / target_len.clamp(min=1.0)).mean()

    def train_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = self.ctc_loss()
        cr_alpha = self.params["training_params"].get("cr_alpha", 0)
        loss_cr = None
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=self.params["training_params"]["use_amp"]):
            if cr_alpha > 0:
                # CR-CTC: two forward passes with different dropout. Both views share the
                # geometric augmentation of the data loader, so their frames stay aligned.
                pred_a = self.models["decoder"](self.models["encoder"](x))
                pred_b = self.models["decoder"](self.models["encoder"](x))
                ctc_a = loss_ctc(pred_a.permute(2, 0, 1), y, x_reduced_len, y_len)
                ctc_b = loss_ctc(pred_b.permute(2, 0, 1), y, x_reduced_len, y_len)
                ctc = 0.5 * (ctc_a + ctc_b)
                loss_cr = self.cr_loss(pred_a, pred_b, x_reduced_len, y_len)
                loss = ctc + cr_alpha * loss_cr
                global_pred = pred_a
            else:
                global_pred = self.models["decoder"](self.models["encoder"](x))
                ctc = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)
                loss = ctc

        pred = torch.argmax(global_pred, dim=1).cpu().numpy()
        if not torch.isfinite(loss):
            print("Non-finite loss, batch skipped: {}".format(batch_data["names"]), flush=True)
            return self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=0.0, loss_cr=0.0, metric_names=metric_names)

        self.backward_loss(loss)

        if self.params["training_params"]["use_amp"]:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # the CTC part only, so that loss_ctc can be compared with runs without CR-CTC
        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=ctc.item(),
                                       loss_cr=0.0 if loss_cr is None else loss_cr.item(), metric_names=metric_names)
        return metrics

    def evaluate_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = self.ctc_loss()

        with torch.no_grad():
            with autocast(device_type="cuda", enabled=self.params["training_params"]["use_amp"]):

                x = self.models["encoder"](x)
                global_pred = self.models["decoder"](x)

                loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)

                if not torch.isfinite(loss):
                    loss = torch.tensor(0.0, device=self.device)

        pred = torch.argmax(global_pred, dim=1).cpu().numpy()
        metrics = self.compute_metrics(pred, y.cpu().numpy(), x_reduced_len, y_len, loss=loss.item(), metric_names=metric_names)
        if "pred" in metric_names:
            metrics["pred"].extend([batch_data["unchanged_labels"], batch_data["names"]])
        return metrics

    def compute_metrics(self, x, y, x_len, y_len, loss=None, loss_cr=None, metric_names=list()):
        batch_size = y.shape[0]
        ind_x = [x[i][:x_len[i]] for i in range(batch_size)]
        ind_y = [y[i][:y_len[i]] for i in range(batch_size)]
        ind_x = [self.ctc_remove_successives_identical_ind(t) for t in ind_x]
        str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in ind_x]
        str_y = [LM_ind_to_str(self.dataset.charset, t) for t in ind_y]
        str_x = [re.sub("( )+", ' ', t).strip(" ") for t in str_x]
        metrics = dict()
        for metric_name in metric_names:
            if metric_name == "cer":
                metrics[metric_name] = [editdistance.eval(u, v) for u,v in zip(str_y, str_x)]
                metrics["nb_chars"] = nb_chars_from_list(str_y)
            elif metric_name == "wer":
                metrics[metric_name] = edit_wer_from_list(str_y, str_x)
                metrics["nb_words"] = nb_words_from_list(str_y)
            elif metric_name == "pred":
                metrics["pred"] = [str_x, ]
        if "loss_ctc" in metric_names:
            # with reduction="mean" the loss is already divided by the number of characters
            if self.params["training_params"].get("ctc_reduction", "sum") == "sum":
                metrics["loss_ctc"] = loss / metrics["nb_chars"]
            else:
                metrics["loss_ctc"] = loss
        if "loss_cr" in metric_names:
            metrics["loss_cr"] = 0.0 if loss_cr is None else loss_cr
        metrics["nb_samples"] = len(x)
        return metrics
