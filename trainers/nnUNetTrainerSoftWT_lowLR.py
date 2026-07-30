from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSoftWT import nnUNetTrainerSoftWT


class nnUNetTrainerSoftWT_lowLR(nnUNetTrainerSoftWT):
    """
    Variante de rescate: mismo mecanismo de soft label en WT, pero con
    learning rate bajo y menos epocas, para un fine-tuning suave desde el 502
    que no degrade el WT ya aprendido.

    Cambios:
      - initial_lr = 1e-4 (en vez de 0.01): fine-tuning suave, preserva lo
        aprendido por el modelo preentrenado.
      - num_epochs = 250: fine-tuning corto (mas barato, suficiente para adaptar).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = 1e-4
        self.num_epochs = 250
