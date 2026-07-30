import torch
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerSoftWT(nnUNetTrainer):
    """
    Fine-tuning con soft labels de Fisher-KPP en Whole Tumor (WT).

    Verificado contra nnU-Net 2.8.1:
      - regions_class_order = [(1,2,3,4), (1,2,3), (1,)] -> canal 0 = WT. OK.
      - _build_loss usa DC_and_BCE_loss + MemoryEfficientSoftDiceLoss,
        que aceptan targets continuos en [0,1]. No hay truncado a entero. OK.
      - build_network_architecture(plans_manager, configuration_manager,
        num_input_channels, num_output_channels, enable_deep_supervision). OK.

    Mecanica:
      1. El dataloader entrega 5 canales: [t1n, t1c, t2w, t2f, soft_wt].
      2. El canal 4 (soft_wt) ha sufrido la MISMA augmentation espacial que las MRI.
      3. train_step: separa el canal 4, lo inyecta como target continuo del WT
         (canal 0 del target) en todas las escalas de deep supervision, y pasa
         solo las 4 MRI a la red.
      4. validation_step: descarta el canal 4 del input y valida contra el
         target DURO original (pseudo-dice real, no adulterado).
      5. La red se fija a 4 canales de entrada -> los pesos del 502 cargan.
    """

    @staticmethod
    def build_network_architecture(plans_manager,
                                   configuration_manager,
                                   num_input_channels,
                                   num_output_channels,
                                   enable_deep_supervision: bool = True):
        # Forzar 4 canales de entrada (ignoramos el 5 que trae el dataloader),
        # para que la arquitectura coincida con los pesos preentrenados del 502.
        return nnUNetTrainer.build_network_architecture(
            plans_manager,
            configuration_manager,
            4,                       # num_input_channels forzado
            num_output_channels,
            enable_deep_supervision
        )

    def initialize(self):
        super().initialize()
        # Coherencia interna: la red ve 4 canales.
        self.num_input_channels = 4

    def _inject_soft_wt(self, batch: dict, hard_target: bool):
        """
        Separa el canal soft del batch. Si hard_target es False (train),
        inyecta el soft en el canal WT (0) del target en todas las escalas.
        Si es True (val), deja el target duro intacto.
        Deja batch['data'] con solo las 4 MRI.
        """
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        mris = data[:, :4]
        soft_wt = data[:, 4:5]  # (B, 1, Z, Y, X)

        if not hard_target:
            if isinstance(target, list):
                for i in range(len(target)):
                    t_shape = target[i].shape[2:]
                    if t_shape != soft_wt.shape[2:]:
                        soft_rescaled = F.interpolate(
                            soft_wt, size=t_shape, mode='trilinear',
                            align_corners=False
                        )
                    else:
                        soft_rescaled = soft_wt
                    # clamp por seguridad numerica tras interpolar
                    target[i][:, 0:1] = soft_rescaled.clamp_(0.0, 1.0)
            else:
                if target.shape[2:] != soft_wt.shape[2:]:
                    soft_rescaled = F.interpolate(
                        soft_wt, size=target.shape[2:], mode='trilinear',
                        align_corners=False
                    )
                else:
                    soft_rescaled = soft_wt
                target[:, 0:1] = soft_rescaled.clamp_(0.0, 1.0)

        batch['data'] = mris
        batch['target'] = target
        return batch

    def train_step(self, batch: dict) -> dict:
        batch = self._inject_soft_wt(batch, hard_target=False)
        return super().train_step(batch)

    def validation_step(self, batch: dict) -> dict:
        # En validacion: input a 4 canales, target DURO (no adulterar el pseudo-dice)
        batch = self._inject_soft_wt(batch, hard_target=True)
        return super().validation_step(batch)
