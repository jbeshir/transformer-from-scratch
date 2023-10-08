from dataclasses import dataclass

import torch as t
import transformer_lens.utils as utils
from torch.utils.data import DataLoader
from torch import Tensor
from transformer_lens import HookedTransformer
import datasets
from typing import Dict
from jaxtyping import Float, Int

import config
import layers

main = __name__ == "__main__"


def lm_cross_entropy_loss(logits: t.Tensor, tokens: t.Tensor):
    # Measure next token loss
    # Logits have shape [batch, position, d_vocab]
    # Tokens have shape [batch, position]
    log_probs = logits.log_softmax(dim=-1)
    pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return -pred_log_probs.mean()


@dataclass
class TransformerTrainingArgs:
    batch_size = 16
    epochs = 10
    max_steps_per_epoch = 200
    lr = 1e-3
    weight_decay = 1e-2


class TransformerTrainer:
    def __init__(self,
                 args: TransformerTrainingArgs,
                 model: layers.Transformer,
                 dataset_dict: datasets.DatasetDict,
                 device):
        super().__init__()
        self.model = model.to(device)
        self.args = args
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.step = 0
        self.dataset_dict = dataset_dict
        self.device = device

    def training_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        '''
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        '''
        inputs = batch["tokens"].to(self.device)
        logits = self.model.forward(inputs)

        loss = lm_cross_entropy_loss(logits, inputs)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    def validation_step(self, batch: Dict[str, Int[Tensor, "batch seq"]]):
        '''
        Calculates & returns the accuracy on the tokens in the batch (i.e. how often the model's prediction
        is correct). Logging should happen in the `train` function (after we've computed the accuracy for
        the whole validation set).
        '''
        inputs = batch["tokens"].to(self.device)
        logits = self.model.forward(inputs)

        loss = lm_cross_entropy_loss(logits, inputs)
        loss.backward()
        return loss

    def train(self):
        '''
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        '''
        train_loader = self.train_loader()

        for i in range(0, self.args.epochs):
            for step, batch in enumerate(train_loader):
                if step >= self.args.max_steps_per_epoch:
                    break
                train_loss = self.training_step(batch)
                print("Training Loss:", train_loss.item())

        test_loader = self.test_loader()
        cumulative_loss = 0
        test_batches = 0
        for _, batch in enumerate(test_loader):
            test_batches += 1
            cumulative_loss += self.validation_step(batch)

        test_loss = cumulative_loss / test_batches
        print("Test Loss:", test_loss.item())

    def train_loader(self) -> DataLoader:
        '''Returns train loader (as in code above).'''
        return DataLoader(self.dataset_dict["train"], batch_size=self.args.batch_size, shuffle=True, num_workers=4,
                          pin_memory=True)

    def test_loader(self) -> DataLoader:
        '''Returns test loader (as in code above).'''
        return DataLoader(self.dataset_dict["test"], batch_size=self.args.batch_size, shuffle=False, num_workers=4,
                          pin_memory=True)


if main:
    # Reuse tokeniser and vocab.
    reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False,
                                                       center_writing_weights=False)

    model_cfg = config.Config(
        debug=False,
        d_model=256,
        n_heads=4,
        d_head=64,
        d_mlp=1024,
        n_layers=2,
        n_ctx=256,
        d_vocab=reference_gpt2.cfg.d_vocab
    )
    model = layers.Transformer(model_cfg)

    args = TransformerTrainingArgs()
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
    print(dataset)
    print(dataset[0]['text'][:100])

    tokenized_dataset = utils.tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False,
                                                       max_length=model_cfg.n_ctx, column_name="text",
                                                       add_bos_token=True,
                                                       num_proc=4)
    dataset_dict = tokenized_dataset.train_test_split(test_size=1000)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    trainer = TransformerTrainer(args, model, dataset_dict, device)
    trainer.train()
