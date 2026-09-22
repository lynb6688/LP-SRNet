"""Reconstruction training and tiled testing for longitudinal super-resolution."""

import torch
from collections import OrderedDict

from basicsr.models.sr_model import SRModel
from basicsr.utils.longitudinal_infer import longitudinal_tile_forward
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class LongitudinalSRModel(SRModel):
    """SMFANet/LP-SRNet reconstruction model with longitudinal tiled inference."""

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            net = self.get_bare_model(self.net_g_ema)
        else:
            self.net_g.eval()
            net = self.get_bare_model(self.net_g)

        tile_size, overlap = _tile_options(self.opt)
        with torch.no_grad():
            self.output = longitudinal_tile_forward(net, self.lq, self.opt['scale'], tile_size, overlap)
        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()


@MODEL_REGISTRY.register()
class TDSRModel(LongitudinalSRModel):
    """Joint reconstruction and frozen-detector fine-tuning.

    ``train.loss_schedule`` is a list of ``{iter, alpha, beta}`` entries. The
    active entry is the last one whose iteration is less than or equal to the
    current iteration. Detector weights are never added to the optimizer.
    """

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']
        self.loss_schedule = list(train_opt.get('loss_schedule', []))
        self.default_alpha = float(train_opt.get('alpha', 1.0))
        self.default_beta = float(train_opt.get('beta', 0.0))
        det_opt = train_opt.get('detector_opt', {})
        weights = self.opt['path'].get('pretrain_network_det', None)
        needs_detector = self.default_beta > 0 or any(float(item.get('beta', 0)) > 0 for item in self.loss_schedule)
        self.net_det = None
        if needs_detector:
            if not weights:
                raise ValueError('A frozen detector checkpoint is required when beta > 0. Set path.pretrain_network_det.')
            from basicsr.models.frozen_detector import FrozenDetector
            self.net_det = FrozenDetector(
                weights,
                cls_weight=det_opt.get('cls_weight', 7.5),
                box_weight=det_opt.get('box_weight', 0.5),
                dfl_weight=det_opt.get('dfl_weight', 1.5),
                imgsz=det_opt.get('imgsz', 640),
            ).to(self.device)
            self.net_det.train()

    def feed_data(self, data):
        super().feed_data(data)
        self.bboxes = data['bboxes'].to(self.device) if 'bboxes' in data else None
        self.det_labels = data['labels'].to(self.device) if 'labels' in data else None
        self.n_targets = data['n_targets'].to(self.device) if 'n_targets' in data else None

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        alpha, beta = self._weights_at(current_iter)

        l_total = 0
        loss_dict = OrderedDict()
        l_rec = self._reconstruction_loss(loss_dict, enable_grad=alpha > 0)
        if alpha > 0:
            l_total = l_total + alpha * l_rec
        if beta > 0:
            if self.net_det is None or self.bboxes is None:
                raise RuntimeError('Detection loss is active, but the frozen detector or GT labels are missing.')
            l_det = self.net_det.detection_loss(self.output, self.det_labels, self.bboxes, self.n_targets)
            l_total = l_total + beta * l_det
            loss_dict['l_det'] = l_det

        if not torch.is_tensor(l_total):
            raise RuntimeError('Both reconstruction and detection weights are zero, so there is nothing to optimize.')

        loss_dict['l_rec'] = l_rec.detach() if torch.is_tensor(l_rec) else self.output.sum() * 0
        loss_dict['alpha'] = self.output.new_tensor(alpha)
        loss_dict['beta'] = self.output.new_tensor(beta)
        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def _reconstruction_loss(self, loss_dict, enable_grad):
        context = torch.enable_grad() if enable_grad else torch.no_grad()
        with context:
            l_rec = self.output.new_zeros(())
            if self.cri_pix:
                l_pix = self.cri_pix(self.output, self.gt)
                l_rec = l_rec + l_pix
                loss_dict['l_pix'] = l_pix
            if self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_rec = l_rec + l_fft
                loss_dict['l_fft'] = l_fft
        return l_rec

    def _weights_at(self, current_iter):
        alpha, beta = self.default_alpha, self.default_beta
        for item in self.loss_schedule:
            if current_iter >= int(item['iter']):
                alpha = float(item['alpha'])
                beta = float(item['beta'])
        return alpha, beta


def _tile_options(opt):
    val_opt = opt.get('val', {}) or {}
    tile_size = val_opt.get('tile_size', None)
    overlap = val_opt.get('tile_overlap', 16)
    return tile_size, overlap
