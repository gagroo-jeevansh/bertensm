import os
import time
from pathlib import Path
import torch
from tqdm import tqdm

def calc_val_loss(model, eval_dataloader, device):
    model.eval()
    val_loss = 0
    val_acc = 0
    val_steps = 0
    val_examples = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = batch.to(device)
            outputs = model(
                **batch,
                # input_ids = batch['input_ids'].to(device),
                # token_type_ids=None,
                # attention_mask=batch['attention_mask'].to(device),
                # labels=batch['labels'].to(device),
            )
            val_loss += outputs.loss.item()
            val_steps += 1
            val_examples += len(batch[0])
            val_acc += outputs.logits.argmax(dim=1).eq(batch[1]).sum().item()
    val_loss /= val_steps
    val_acc /= val_examples
    model.train()
    return val_loss, val_acc

def train(args, model, train_dataloader, eval_dataloader, optimizer, device, stats_path):
    train_losses = []
    num_all_pts = 0
    train_losses = []
    val_losses = []
    val_accs = []

    stats_path = os.path.join(args.savepath, "stats")
    Path(stats_path).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    model.to(device)
    
    num_steps = args.epochs * len(train_dataloader)
    
    progress_bar = tqdm(range(num_steps))

    model.train()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        tr_examples, tr_steps = 0, 0

        print(f"=======>Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch = batch.to(device)
            outputs = model(
                **batch,
                # input_ids = batch['input_ids'].to(device),
                # token_type_ids=None,
                # attention_mask=batch['attention_mask'].to(device),
                # labels=batch['labels'].to(device),
            )
            outputs.loss.backward()
            optimizer.step()
            progress_bar.update(1)
            train_loss += outputs.loss.item()
            tr_examples += len(batch[0])
            num_all_pts += len(batch[0])
            tr_steps += 1
            train_losses.append(train_loss / tr_steps)

            time_elapsed = (time.time() - start_time) / 60

            # Validation Loss
            val_loss, val_acc = calc_val_loss(model, eval_dataloader, device)
            print(
                f"Epoch: {epoch+1}/{args.epochs} | Time Elapsed: {time_elapsed:.2f} mins | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
            val_losses.append(val_loss)
            val_accs.append(val_acc)

    return train_losses, val_losses, val_accs